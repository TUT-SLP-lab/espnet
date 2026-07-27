#!/usr/bin/env python3
"""Create a readable per-utterance comparison of inter-CTC layer outputs."""

from __future__ import annotations

import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


DEFAULT_MAIN_RECOG_DIR = (
    SCRIPT_DIR
    / "exp/asr_interCTC_conv2d6/decode_ctc_bs1_asr_model_valid.cer_ctc.ave"
    / "test_clean/logdir/output.1/1best_recog"
)
DEFAULT_AUX_RECOG_DIR = (
    SCRIPT_DIR
    / "exp/asr_atten_loss_conv2d6/decode_ctc_bs1_asr_model_valid.cer_ctc.ave"
    / "test_clean/logdir/output.1/1best_recog"
)
DEFAULT_REF_TRN = (
    SCRIPT_DIR
    / "exp/asr_interCTC_conv2d6/decode_ctc_bs1_asr_model_valid.cer_ctc.ave"
    / "test_clean/score_wer/ref.trn"
)
DEFAULT_OUTPUT_NAME = "combined_interctc_layers_3_15_atten_loss_conv2d6_19_20_vertical.txt"


def normalize_hyp(text: str) -> str:
    return text.replace(" ", "").replace("▁", " ").strip()


def read_recog(path: Path) -> tuple[list[str], dict[str, str]]:
    rows: dict[str, str] = {}
    order: list[str] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt = parts[0]
            hyp = normalize_hyp(parts[1] if len(parts) > 1 else "")
            if utt in rows:
                raise ValueError(f"duplicate utterance id in {path}:{lineno}: {utt}")
            rows[utt] = hyp
            order.append(utt)
    return order, rows


def read_refs(path: Path) -> dict[str, str]:
    refs_by_trn_id: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                text, trn_id = line.rsplit("\t", 1)
            else:
                text, trn_id = line.rsplit(maxsplit=1)
            trn_id = trn_id.strip()
            if not (trn_id.startswith("(") and trn_id.endswith(")")):
                raise ValueError(f"unexpected ref.trn id format at {path}:{lineno}: {trn_id}")
            refs_by_trn_id[trn_id[1:-1]] = text
    return refs_by_trn_id


def ref_for_utt(refs_by_trn_id: dict[str, str], utt: str) -> str:
    if utt in refs_by_trn_id:
        return refs_by_trn_id[utt]

    matches = [text for trn_id, text in refs_by_trn_id.items() if trn_id.endswith(utt)]
    if len(matches) != 1:
        raise ValueError(f"expected one ref for {utt}, got {len(matches)}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge ref.trn and encoder_interctc_layer*.txt files into a readable "
            "per-utterance vertical comparison."
        )
    )
    parser.add_argument("--main-recog-dir", type=Path, default=DEFAULT_MAIN_RECOG_DIR)
    parser.add_argument("--aux-recog-dir", type=Path, default=DEFAULT_AUX_RECOG_DIR)
    parser.add_argument("--ref-trn", type=Path, default=DEFAULT_REF_TRN)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to MAIN_RECOG_DIR/combined_interctc_layers_...txt",
    )
    parser.add_argument("--main-layers", type=int, nargs="+", default=[3, 6, 9, 12, 15])
    parser.add_argument("--aux-layers", type=int, nargs="+", default=[19, 20])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.main_recog_dir / DEFAULT_OUTPUT_NAME

    layer_files = [
        (f"layer{layer}", args.main_recog_dir / f"encoder_interctc_layer{layer}.txt")
        for layer in args.main_layers
    ]
    layer_files.extend(
        (f"layer{layer}", args.aux_recog_dir / f"encoder_interctc_layer{layer}.txt")
        for layer in args.aux_layers
    )

    refs_by_trn_id = read_refs(args.ref_trn)
    first_order: list[str] | None = None
    all_rows: dict[str, dict[str, str]] = {}

    for label, path in layer_files:
        order, rows = read_recog(path)
        if first_order is None:
            first_order = order
        elif set(order) != set(first_order):
            missing = sorted(set(first_order) - set(order))[:5]
            extra = sorted(set(order) - set(first_order))[:5]
            raise ValueError(f"utterance ids differ for {label}: missing={missing}, extra={extra}")
        all_rows[label] = rows

    if first_order is None:
        raise ValueError("no layer files were specified")

    labels = [label for label, _ in layer_files]
    with output.open("w", encoding="utf-8") as f:
        for i, utt in enumerate(first_order):
            if i:
                f.write("\n")
            f.write(f"{utt}\n")
            f.write(f"ref: {ref_for_utt(refs_by_trn_id, utt)}\n")
            for label in labels:
                f.write(f"{label}: {all_rows[label][utt]}\n")

    print(output)
    print(f"wrote {len(first_order)} utterance blocks with ref and normalized layer text")


if __name__ == "__main__":
    main()
