import argparse
from typing import Optional
import pyopenjtalk  # type: ignore

# import MeCab  # type: ignore


# tagger = MeCab.Tagger("-Oyomi")
# tagger.parse("")  # avoid bug


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    # parser.add_argument("--output_chars", type=str, required=True)
    parser.add_argument("--output_error", type=str, required=True)
    parser.add_argument("--output4train", type=str, default=None)
    parser.add_argument("--field", type=str, default="2-")
    parser.add_argument("--delimiter", type=str, default=" ")
    parser.add_argument("--joint_symbol", type=str, default="@")
    return parser.parse_args()


def field2slice(field: Optional[str]) -> slice:
    """Convert field string to slice.
    Note that field string accepts 1-based integer.
    Examples:
        >>> field2slice("1-")
        slice(0, None, None)
        >>> field2slice("1-3")
        slice(0, 3, None)
        >>> field2slice("-3")
        slice(None, 3, None)
    """
    field = field.strip()
    try:
        if "-" in field:
            # e.g. "2-" or "2-5" or "-7"
            s1, s2 = field.split("-", maxsplit=1)
            if s1.strip() == "":
                s1 = None
            else:
                s1 = int(s1)
                if s1 == 0:
                    raise ValueError("1-based string")
            if s2.strip() == "":
                s2 = None
            else:
                s2 = int(s2)
        else:
            # e.g. "2"
            s1 = int(field)
            s2 = s1 + 1
            if s1 == 0:
                raise ValueError("must be 1 or more value")
    except ValueError:
        raise RuntimeError(f"Format error: e.g. '2-', '2-5', or '-5': {field}")

    if s1 is None:
        slic = slice(None, s2)
        anti_slic = slice(s2, None)
    else:
        # -1 because of 1-based integer following "cut" command
        # e.g "1-3" -> slice(0, 3)
        slic = slice(s1 - 1, s2)
        anti_slic = slice(None, s1 - 1)
    return [anti_slic, slic]


args = parse_args()
_cils, _slic = field2slice(args.field)

with open(args.input, "r", encoding="utf-8") as fin:
    # fout_c = open(args.output_chars, "w", encoding="utf-8")
    fout_p = open(args.output, "w", encoding="utf-8")
    fout_e = open(args.output_error, "a", encoding="utf-8")
    fout_t = (
        open(args.output4train, "w", encoding="utf-8")
        if args.output4train != "NONE"
        else None
    )

    for line in fin:
        textinfo = line.rstrip()
        text = textinfo.split(args.delimiter)[_slic]
        info = textinfo.split(args.delimiter)[_cils]
        text: str = args.delimiter.join(text)
        info: str = args.delimiter.join(info)
        text = text.strip(" 　")
        info = info.strip(" 　")
        text = text.replace("・", "")
        text = text.replace("−", "")

        # katakana = tagger.parse(text).strip()
        phones = pyopenjtalk.g2p(text)

        # if "pau" in phones:
        #     fout_e.write(args.delimiter.join([textinfo, phones]) + "\n")
        #     continue

        output_text = args.delimiter.join([info, phones])
        fout_p.write(output_text + "\n")
        # fout_c.write(textinfo + "\n")
        if fout_t:
            texts = args.joint_symbol.join([textinfo, phones])
            fout_t.write(texts + "\n")

    # fout_c.close()
    fout_p.close()
    fout_e.close()
    if fout_t:
        fout_t.close()