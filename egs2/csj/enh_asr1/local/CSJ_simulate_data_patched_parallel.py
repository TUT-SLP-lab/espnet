import logging
import glob
from multiprocessing.sharedctypes import Value
import numpy as np
import argparse
import os
import shutil
from joblib import Parallel, delayed
from scipy.io.wavfile import read, write
import random


logging.basicConfig(level=logging.DEBUG)


class NoiseSuperimposition:
    def __init__(self, nj: int, splited_wav_dir: str, noise_path: str, data_type: str, dist: str):
        """
        nj : nuber jobs
        csj_path : path of CSJ
        noise_path : path to CHiME4 or CHiME3 or other noises
        """
        self.nj = nj
        self.splited_wav_dir = splited_wav_dir
        self.noise_path = noise_path
        self.data_type = data_type

        self.dist_noisy = dist + "/noisy"
        self.dist_isolated = dist + "/isolated"
        # make directory
        if os.path.exists(self.dist_noisy):
            shutil.rmtree(self.dist_noisy)
        if os.path.exists(self.dist_isolated):
            shutil.rmtree(self.dist_isolated)
        os.makedirs(self.dist_noisy)
        os.makedirs(self.dist_isolated)

        self.DEBUG = False
    @staticmethod
    def cal_rms(amp):
        mean = np.mean(np.square(amp))
        return np.sqrt(mean)

    @staticmethod
    def cal_adjusted_rms(clean_rms, snr):
        a = float(snr)/20
        noise_rms = clean_rms/(10**a)
        return noise_rms

    def superimposition(self, csj_file: str, noise_file: str, snr: float):
        """雑音重畳"""

        # clean data
        clean_rate, clean_data = read(csj_file)
        if self.DEBUG:
            logging.info("===clean data===")
            logging.info(f"sampling rate: {clean_rate} data: {clean_data} len: {len(clean_data)}")

        # noise data
        noise_rate, noise_data = read(noise_file)
        if self.DEBUG:
            logging.info("===noise data===")
            logging.info(f"sampling rate: {noise_rate} data: {noise_data} len: {len(noise_data)}")

        # test set はfirst, validation set はend
        half = int(len(noise_data) / 2)
        if self.data_type == "test":
            print("use first")
            noise_data = noise_data[:half]
        if self.data_type == "valid":
            print("use end")
            noise_data = noise_data[half:]

        # # cleanデータのほうが長かったら、noiseを繰り返す。
        # while len(clean_data) > len(noise_data):
        #     noise_data = np.concatenate([noise_data, noise_data])

        # # モノラルなら無視(-R.wavと-L.wavがあるから)
        # if clean_data.ndim == 2:
        #     print("this is monoral")
        #     return

        # ランダム抽出＋同じ長さにして重畳
        noise_start_index = 0
        if self.data_type == "train":
            noise_start_index = random.randint(0, len(noise_data)-len(clean_data))
        else:
            # test, validのときは0~に固定
            noise_start_index = 0
        noise_data = noise_data[noise_start_index : noise_start_index + len(clean_data)]

        # Root Mean Square RMSを求める
        clean_rms = self.cal_rms(clean_data)
        noise_rms = self.cal_rms(noise_data)

        # snrに対応したrmsを求める
        adjusted_noise_rms = self.cal_adjusted_rms(clean_rms, snr)
        adjusted_noise_data = noise_data * (adjusted_noise_rms/ noise_rms)
        adjusted_noise_data = adjusted_noise_data.astype(np.int16)# intに変換

        # 雑音重畳
        noisy_data = clean_data + adjusted_noise_data
        if clean_rate != noise_rate:
            logging.error("sampling rate is different")
            return

        # 正規化　(wavが16bitなので、符号をどけた2^15 ~ -2^15の値に正規化)
        max_value = np.abs(noisy_data).max()
        if max_value > 32767:
            noisy_data = noisy_data * (32767/max_value)
            adjusted_noise_data = adjusted_noise_data * (32767/max_value)
            clean_data = clean_data * (32767/max_value)


        # write noisy data
        noise_name = noise_file.split("_")[-1].split(".")  # [STR , CH0]
        noise_no = noise_file.split("_")[-2]  # 020
        csj_name = csj_file.split("/")[-1].split(".")[0]  # A01M0056_00000_00000

        return_value =[]
        # <dist_path>/A01M0056_STR.CH0.wav
        utt_name = f"{csj_name}_{noise_no}_{noise_name[0]}_{noise_name[1]}_{int(snr)}"
        dist_file = f"{self.dist_noisy   }/{utt_name}_SIMU.wav"
        write(dist_file, noise_rate, noisy_data)
        return_value.append(f"{utt_name}_SIMU {dist_file}\n")

        # output clean file
        dist_file = f"{self.dist_isolated}/{utt_name}.Clean.wav"
        write(dist_file, noise_rate, clean_data)
        return_value.append(f"{utt_name} {dist_file}\n")

        # output noise file
        dist_file = f"{self.dist_isolated}/{utt_name}.Noise.wav"
        write(dist_file, noise_rate, adjusted_noise_data)
        return_value.append(f"{utt_name} {dist_file}\n")

        return return_value

    def run_parallel(self):

        # get wav path
        csj_path_list = []
        csj_path_list = glob.glob(self.splited_wav_dir+"/*.wav", recursive=True) # aaaaaa_1234_1234.wav
        noise_path_list = glob.glob(self.noise_path + "/*.wav", recursive=True)

        print(self.splited_wav_dir+"/*.wav")
        print(f"csj list sample:{csj_path_list} csj list mount:{len(csj_path_list)}")
        print(f"noise list sample: {noise_path_list} noise list mount:{len(noise_path_list)}")

        print("make simulation")


        wav_list = []
        # make core simulation
        # if self.data_type=="train":
        # snr is random
        for n in noise_path_list:
            result = Parallel(n_jobs=self.nj)([delayed(self.superimposition)(c, n, random.random()*20) for c in csj_path_list])
            wav_list.extend(result)
            print(f"finished to simulate {n}")
        # else:
        #     # test and valid 
        #     for n in noise_path_list:
        #         for snr in [5, 0, -5]: # 5db, 0db, -5dbで作成
        #             result = Parallel(n_jobs=self.nj)([delayed(self.superimposition)(c, n, snr) for c in csj_path_list])
        #             wav_list.extend(result)
        #             print(f"finished to simulate {n}")

        wav_lines   = [r[0] for r in wav_list]
        clean_lines = [r[1] for r in wav_list]
        noise_lines = [r[2] for r in wav_list]

        with open(f"{self.dist_noisy}/wav.scp", "w") as f:
            f.writelines(wav_lines)
        with open(f"{self.dist_isolated}/spk1.scp", "w") as f:
            f.writelines(clean_lines)
        with open(f"{self.dist_isolated}/noise1.scp", "w") as f:
            f.writelines(noise_lines)
        print("finish to write scp")
        

def get_args():
    parser = argparse.ArgumentParser(description="Simulate data with parallel process")
    parser.add_argument(
        "--splited_wav_dir", type=str, required=True, help="path to wav file splited",
    )
    parser.add_argument(
        "--noise_path", type=str, required=True, help="path to noise directory",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["train", "valid", "test"],
        default="train",
        help="what type of data?, 'train', 'valid', 'test",
    )
    parser.add_argument(
        "--dist", type=str, required=True, help="path to noisy directory",
    )
    parser.add_argument(
        "--nj", type=int, default=0, required=True, help="number of jobs",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    ns = NoiseSuperimposition(args.nj, args.splited_wav_dir, args.noise_path, args.data_type, args.dist)
    ns.run_parallel()

    # ns = NoiseSuperimposition(1, "", "args.noise_path", "train", ".")
    # ns.superimposition("./A00000_000_000_clean.wav", "./A00000_000_000_noise.CH0.wav", 5)
