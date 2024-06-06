import json
import logging
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Optional, Tuple, Union

import librosa
import numpy as np
import scipy.signal
import soundfile
from typeguard import check_argument_types, check_return_type

from espnet2.layers.augmentation import DataAugmentation
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.cleaner import TextCleaner
from espnet2.text.token_id_converter import TokenIDConverter

from espnet2.train.preprocessor import AbsPreprocessor


def framing(
    x,
    frame_length: int = 512,
    frame_shift: int = 256,
    centered: bool = True,
    padded: bool = True,
):
    if x.size == 0:
        raise ValueError("Input array size is zero")
    if frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if frame_length > x.shape[-1]:
        raise ValueError("frame_length is greater than input length")
    if 0 >= frame_shift:
        raise ValueError("frame_shift must be greater than 0")

    if centered:
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
            (frame_length // 2, frame_length // 2)
        ]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
        #  with integer nseg
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    # Created strided array of data segments
    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., None]
    else:
        shape = x.shape[:-1] + (
            (x.shape[-1] - frame_length) // frame_shift + 1,
            frame_length,
        )
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result


def detect_non_silence(
    x: np.ndarray,
    threshold: float = 0.01,
    frame_length: int = 1024,
    frame_shift: int = 512,
    window: str = "boxcar",
) -> np.ndarray:
    """Power based voice activity detection.

    Args:
        x: (Channel, Time)
    >>> x = np.random.randn(1000)
    >>> detect = detect_non_silence(x)
    >>> assert x.shape == detect.shape
    >>> assert detect.dtype == np.bool
    """
    if x.shape[-1] < frame_length:
        return np.full(x.shape, fill_value=True, dtype=np.bool)

    if x.dtype.kind == "i":
        x = x.astype(np.float64)
    # framed_w: (C, T, F)
    framed_w = framing(
        x,
        frame_length=frame_length,
        frame_shift=frame_shift,
        centered=False,
        padded=True,
    )
    framed_w *= scipy.signal.get_window(window, frame_length).astype(framed_w.dtype)
    # power: (C, T)
    power = (framed_w**2).mean(axis=-1)
    # mean_power: (C, 1)
    mean_power = np.mean(power, axis=-1, keepdims=True)
    if np.all(mean_power == 0):
        return np.full(x.shape, fill_value=True, dtype=np.bool)
    # detect_frames: (C, T)
    detect_frames = power / mean_power > threshold
    # detects: (C, T, F)
    detects = np.broadcast_to(
        detect_frames[..., None], detect_frames.shape + (frame_shift,)
    )
    # detects: (C, TF)
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    # detects: (C, TF)
    return np.pad(
        detects,
        [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
        mode="edge",
    )


def any_allzero(signal):
    if isinstance(signal, (list, tuple)):
        return any([np.allclose(s, 0.0) for s in signal])
    return np.allclose(signal, 0.0)


class ASRPPreprocessor(AbsPreprocessor):
    def __init__(
        self,
        train: bool,
        use_lang_prompt: bool = False,
        use_nlp_prompt: bool = False,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        phone_token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        phone_non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        joint_symbol: str = "@",
        delimiter: str = None,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        aux_task_names: Collection[str] = None,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: str = "text",
        phoneme_name: str = "phoneme",
        fs: int = 0,
        nonsplit_symbol: Iterable[str] = None,
        data_aug_effects: List = None,
        data_aug_num: List[int] = [1, 1],
        data_aug_prob: float = 0.0,
        # only use for whisper
        whisper_language: str = None,
        whisper_task: str = None,
    ):
        super().__init__(train)
        self.train = train
        self.speech_name = speech_name
        self.text_name = text_name
        self.phoneme_name = phoneme_name
        self.speech_volume_normalize = speech_volume_normalize
        self.rir_apply_prob = rir_apply_prob
        self.noise_apply_prob = noise_apply_prob
        self.short_noise_thres = short_noise_thres
        self.aux_task_names = aux_task_names
        self.use_lang_prompt = use_lang_prompt
        self.use_nlp_prompt = use_nlp_prompt

        # if token_type == "aux_phone" or token_type == "char_phone":
        if token_type is not None:
            if token_list is None:
                raise ValueError("token_list is required if token_type is not None")

            self.text_cleaner = TextCleaner(text_cleaner)

            self.tokenizer = build_tokenizer(
                token_type=token_type,
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
                nonsplit_symbol=nonsplit_symbol,
                remove_non_linguistic_symbols=False,
            )

            self.phone_tokenizer = build_tokenizer(
                token_type="phn",
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=phone_non_linguistic_symbols,
                g2p_type=g2p_type,
                remove_non_linguistic_symbols=False,
            )
            self.phone_tokenizer = build_tokenizer(
                token_type="word",
                delimiter=" ",
                non_linguistic_symbols=phone_non_linguistic_symbols,
                remove_non_linguistic_symbols=False,
            )

            self.token_id_converter = TokenIDConverter(
                token_list=token_list,
                unk_symbol=unk_symbol,
            )
            self.phone_token_id_converter = TokenIDConverter(
                token_list=phone_token_list,
                unk_symbol=unk_symbol,
            )
        else:
            self.text_cleaner = None
            self.tokenizer = None
            self.char_token_id_converter = None
            self.phone_token_id_converter = None

        if train and rir_scp is not None:
            self.rirs = []
            rir_scp = [rir_scp] if not isinstance(rir_scp, (list, tuple)) else rir_scp
            for scp in rir_scp:
                with open(scp, "r", encoding="utf-8") as f:
                    for line in f:
                        sps = line.strip().split(None, 1)
                        if len(sps) == 1:
                            self.rirs.append(sps[0])
                        else:
                            self.rirs.append(sps[1])
        else:
            self.rirs = None

        if train and noise_scp is not None:
            self.noises = []
            noise_scp = (
                [noise_scp] if not isinstance(noise_scp, (list, tuple)) else noise_scp
            )
            for scp in noise_scp:
                with open(scp, "r", encoding="utf-8") as f:
                    for line in f:
                        sps = line.strip().split(None, 1)
                        if len(sps) == 1:
                            self.noises.append(sps[0])
                        else:
                            self.noises.append(sps[1])
            sps = noise_db_range.split("_")
            if len(sps) == 1:
                self.noise_db_low = self.noise_db_high = float(sps[0])
            elif len(sps) == 2:
                self.noise_db_low, self.noise_db_high = float(sps[0]), float(sps[1])
            else:
                raise ValueError(
                    "Format error: '{noise_db_range}' e.g. -3_4 -> [-3db,4db]"
                )
        else:
            self.noises = None

        # Check DataAugmentation docstring for more information of `data_aug_effects`
        self.fs = fs
        if data_aug_effects is not None:
            assert self.fs > 0, self.fs
            self.data_aug = DataAugmentation(data_aug_effects, apply_n=data_aug_num)
        else:
            self.data_aug = None
        self.data_aug_prob = data_aug_prob

    def _convolve_rir(self, speech, power, rirs, tgt_fs=None, single_channel=False):
        rir_path = np.random.choice(rirs)
        rir = None
        if rir_path is not None:
            rir, fs = soundfile.read(rir_path, dtype=np.float64, always_2d=True)

            if single_channel:
                num_ch = rir.shape[1]
                chs = [np.random.randint(num_ch)]
                rir = rir[:, chs]
            # rir: (Nmic, Time)
            rir = rir.T
            if tgt_fs and fs != tgt_fs:
                logging.warning(
                    f"Resampling RIR to match the sampling rate ({fs} -> {tgt_fs} Hz)"
                )
                rir = librosa.resample(
                    rir, orig_sr=fs, target_sr=tgt_fs, res_type="kaiser_fast"
                )

            # speech: (Nmic, Time)
            speech = speech[:1]
            # Note that this operation doesn't change the signal length
            speech = scipy.signal.convolve(speech, rir, mode="full")[
                :, : speech.shape[1]
            ]
            # Reverse mean power to the original power
            power2 = (speech[detect_non_silence(speech)] ** 2).mean()
            speech = np.sqrt(power / max(power2, 1e-10)) * speech
        return speech, rir

    def _add_noise(
        self,
        speech,
        power,
        noises,
        noise_db_low,
        noise_db_high,
        tgt_fs=None,
        single_channel=False,
    ):
        nsamples = speech.shape[1]
        noise_path = np.random.choice(noises)
        noise = None
        if noise_path is not None:
            noise_db = np.random.uniform(noise_db_low, noise_db_high)
            with soundfile.SoundFile(noise_path) as f:
                fs = f.samplerate
                if tgt_fs and fs != tgt_fs:
                    nsamples_ = int(nsamples / tgt_fs * fs) + 1
                else:
                    nsamples_ = nsamples
                if f.frames == nsamples_:
                    noise = f.read(dtype=np.float64, always_2d=True)
                elif f.frames < nsamples_:
                    if f.frames / nsamples_ < self.short_noise_thres:
                        logging.warning(
                            f"Noise ({f.frames}) is much shorter than "
                            f"speech ({nsamples_}) in dynamic mixing"
                        )
                    offset = np.random.randint(0, nsamples_ - f.frames)
                    # noise: (Time, Nmic)
                    noise = f.read(dtype=np.float64, always_2d=True)
                    # Repeat noise
                    noise = np.pad(
                        noise,
                        [(offset, nsamples_ - f.frames - offset), (0, 0)],
                        mode="wrap",
                    )
                else:
                    offset = np.random.randint(0, f.frames - nsamples_)
                    f.seek(offset)
                    # noise: (Time, Nmic)
                    noise = f.read(nsamples_, dtype=np.float64, always_2d=True)
                    if len(noise) != nsamples_:
                        raise RuntimeError(f"Something wrong: {noise_path}")
            if single_channel:
                num_ch = noise.shape[1]
                chs = [np.random.randint(num_ch)]
                noise = noise[:, chs]
            # noise: (Nmic, Time)
            noise = noise.T
            if tgt_fs and fs != tgt_fs:
                logging.warning(
                    f"Resampling noise to match the sampling rate ({fs} -> {tgt_fs} Hz)"
                )
                noise = librosa.resample(
                    noise, orig_sr=fs, target_sr=tgt_fs, res_type="kaiser_fast"
                )
                if noise.shape[1] < nsamples:
                    noise = np.pad(
                        noise, [(0, 0), (0, nsamples - noise.shape[1])], mode="wrap"
                    )
                else:
                    noise = noise[:, :nsamples]

            noise_power = (noise**2).mean()
            scale = (
                10 ** (-noise_db / 20)
                * np.sqrt(power)
                / np.sqrt(max(noise_power, 1e-10))
            )
            speech = speech + scale * noise
        return speech, noise

    def _speech_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        assert check_argument_types()
        if self.speech_name in data:
            if self.train and (self.rirs is not None or self.noises is not None):
                speech = data[self.speech_name]

                # speech: (Nmic, Time)
                if speech.ndim == 1:
                    speech = speech[None, :]
                else:
                    speech = speech.T
                # Calc power on non silence region
                power = (speech[detect_non_silence(speech)] ** 2).mean()

                # 1. Convolve RIR
                if self.rirs is not None and self.rir_apply_prob >= np.random.random():
                    speech, _ = self._convolve_rir(speech, power, self.rirs)

                # 2. Add Noise
                if (
                    self.noises is not None
                    and self.noise_apply_prob >= np.random.random()
                ):
                    speech, _ = self._add_noise(
                        speech,
                        power,
                        self.noises,
                        self.noise_db_low,
                        self.noise_db_high,
                    )

                speech = speech.T
                ma = np.max(np.abs(speech))
                if ma > 1.0:
                    speech /= ma
                data[self.speech_name] = speech

            if self.train and self.data_aug:
                if self.data_aug_prob > 0 and self.data_aug_prob >= np.random.random():
                    data[self.speech_name] = self.data_aug(
                        data[self.speech_name], self.fs
                    )

            if self.speech_volume_normalize is not None:
                speech = data[self.speech_name]
                ma = np.max(np.abs(speech))
                data[self.speech_name] = speech * self.speech_volume_normalize / ma
        assert check_return_type(data)
        return data

    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if self.text_name in data and self.tokenizer is not None:
            text = data[self.text_name]
            if isinstance(text, np.ndarray):
                return data
            text = self.text_cleaner(text)
            tokens = self.tokenizer.text2tokens(text)
            text_ints = self.token_id_converter.tokens2ids(tokens)
            if len(text_ints) > 500:
                logging.warning(
                    "The length of the text output exceeds 500, "
                    "which may cause OOM on the GPU."
                    "Please ensure that the data processing is correct and verify it."
                )
            data[self.text_name] = np.array(text_ints, dtype=np.int64)
            if "prompt" in data:
                raise NotImplementedError("prompt is not supported ASRPreprocessor")
        if self.phoneme_name in data and self.phone_tokenizer is not None:
            text = data[self.phoneme_name]
            if isinstance(text, np.ndarray):
                return data
            text = self.text_cleaner(text)
            tokens = self.phone_tokenizer.text2tokens(text)
            text_ints = self.phone_token_id_converter.tokens2ids(tokens)
            if len(text_ints) > 500:
                logging.warning(
                    "The length of the text output exceeds 500, "
                    "which may cause OOM on the GPU."
                    "Please ensure that the data processing is correct and verify it."
                )
            data[self.phoneme_name] = np.array(text_ints, dtype=np.int64)
            if "prompt" in data:
                raise NotImplementedError("prompt is not supported ASRPreprocessor")
        if self.aux_task_names is not None and self.tokenizer is not None:
            for name in self.aux_task_names:
                if name == "phoneme":
                    text = data[name]
                    text = self.text_cleaner(text)
                    tokens = self.tokenizer.phone_text2tokens(text)
                    text_ints = self.phone_token_id_converter.tokens2ids(tokens)
                    data[name] = np.array(text_ints, dtype=np.int64)
                if name in data:
                    text = data[name]
                    text = self.text_cleaner(text)
                    tokens = self.tokenizer.text2tokens(text)
                    text_ints = self.char_token_id_converter.tokens2ids(tokens)
                    data[name] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        data = self._speech_process(data)
        data = self._text_process(data)
        return data