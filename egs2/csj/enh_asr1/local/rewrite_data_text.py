"""
# dataのパスが変わったので、data/*のなかのいくつかを書き換えなければいけない。
# segment A01M0097_010_BUS.CH1.0000211_0001695 A01M0097 0.211 1.695
# uut2spk (後にutil/utt2spk_to_spk2utt.pl) (segmentを使えばいけそう)
# text 
# 他足りないもの
# utt2category
# noise1.scp
# spk1.scp (Clean.wavのパスを入れてやればいい。)

"""

import argparse
import glob
from joblib import Parallel, delayed


class RewriteDataText:
    def __init__(self, noisy_path: str, isolated_path: str, data_path: str) -> None:
        self.noisy_path = noisy_path
        self.isolated_path = isolated_path
        self.data_path = data_path

        self.noisy_wav_list = glob.glob(self.noisy_path + "/*.wav")
        self.noise_wav_list = glob.glob(self.isolated_path + "/*Noise.wav")
        self.clean_wav_list = glob.glob(self.isolated_path + "/*Clean.wav")
        # backup

    def rewrite_segments(self):
        # parallel 処理
        def segment_replace(search_name, file_name, s):
            return s.replace(search_name, file_name.split(".")[0])

        with open(self.data_path + "/segments") as f:
            lines = f.readlines()

            # prepare segment dict
            line_list = []
            current_name = ""
            segmtn_dict = {}
            for line in lines:
                data_name = line.split(" ")[0].split("_")[0]  # A01M0097 from segment
                if current_name != data_name:
                    segmtn_dict[current_name] = line_list
                    line_list = []
                    current_name = data_name
                line_list.append(line)

            segmtn_dict[current_name] = line_list
            print("prepared segment")

            line_list = []
            for i, file_name in enumerate(self.noisy_wav_list):
                file_name = file_name.split("/")[-1]
                search_name = file_name.split("-")[0]  # A02M0012 from noisy file
                result_line = Parallel(n_jobs=-1)(
                    delayed(segment_replace)(search_name, file_name, s) for s in segmtn_dict[search_name]
                )
                # 結合
                line_list += result_line
                if i % 100 == 0:
                    print(f"finish to prepare {i+1}/{len(self.noisy_wav_list)}")

            line_list.sort()

        with open(self.data_path + "/segments", "w") as f:  # 上書き
            f.writelines(line_list)

        print("finish to rewrite segment!")

    def rewrite_spk1(self):
        # rewrite to clean.wav
        lines = []
        for clean in self.clean_wav_list:
            wav_name = clean.split("/")[-1].split(".")[0]  # A01M0141-010-CAF
            lines.append(f"{wav_name} cat {clean} |\n")
        lines.sort()
        with open(self.data_path + "/spk1.scp", "w") as f:
            f.writelines(lines)

        print("finish to rewrite spk1!")

    def rewrite_noise1(self):
        # rewrite to Noise.wav
        lines = []
        for clean in self.noise_wav_list:
            file_name = clean.split("/")[-1].split(".")[0]  # A01M0141-010-CAF
            lines.append(f"{file_name} cat {clean} |\n")

        lines.sort()
        with open(self.data_path + "/noise1.scp", "w") as f:
            f.writelines(lines)
        print("finish to rewrite noise1!")

    def rewrite_wav(self):
        # rewrite to Noise.wav
        lines = []
        for wav in self.noisy_wav_list:
            file_name = wav.split("/")[-1].split(".")[0]
            lines.append(f"{file_name} cat {wav} |\n")

        lines.sort()
        with open(self.data_path + "/wav.scp", "w") as f:
            f.writelines(lines)
        print("finish to rewrite wav!")

    def rewrite_text_spk2utt(self):

        spk2utt_list = [""]
        with open(self.data_path + "/segments") as f:
            current_data = ""
            for line in f:
                segment_name = line.split(" ")[0]
                name = segment_name.split("_")[0].split("-")[0]  # A05M0011
                if current_data != name:
                    current_data = name
                    spk2utt_list[-1] = spk2utt_list[-1] + "\n"
                    spk2utt_list.append(name)
                spk2utt_list[-1] = spk2utt_list[-1] + " " + segment_name
        # ""を除去
        spk2utt_list = spk2utt_list[1:]

        spk2utt_list.sort()

        # utt2spk
        with open(self.data_path + "/spk2utt", "w") as f:
            f.writelines(spk2utt_list)
        print("finish to rewrite spk2ut")

    def rewrite_text(self):
        # parallel　処理
        def text_replace(search_name, file_name, s):
            return s.replace(search_name, file_name.split(".")[0])

        with open(self.data_path + "/text") as f:
            lines = f.readlines()

            # prepare segment dict
            line_list = []
            current_name = ""
            text_dict = {}

            for line in lines:
                data_name = line.split(" ")[0].split("_")[0]  # A01M0097

                if current_name != data_name:
                    text_dict[current_name] = line_list
                    line_list = []
                    current_name = data_name
                line_list.append(line)

            text_dict[current_name] = line_list
            print("prepared text dict")

            line_list = []
            for i, file_name in enumerate(self.noisy_wav_list):
                file_name = file_name.split("/")[-1]
                search_name = file_name.split("-")[0]
                result_line = Parallel(n_jobs=-1)(
                    delayed(text_replace)(search_name, file_name, s) for s in text_dict[search_name]
                )
                # 結合
                line_list += result_line
                if i % 100 == 0:
                    print(f"finish to prepare {i+1}/{len(self.noisy_wav_list)}")
            line_list.sort()

        with open(self.data_path + "/text", "w") as f:  # 上書き
            f.writelines(line_list)

        print("finish to rewrite text!")

    def rewrite_utt2category(self):
        pass


def get_args():
    parser = argparse.ArgumentParser(description="rewrite the data text")

    parser.add_argument(
        "--noisy-path", type=str, required=True, help="path to scp_file",
    )
    parser.add_argument(
        "--isolated-path", type=str, required=True, help="path to noise directory",
    )
    parser.add_argument("--data-path", type=str, required=True, help="path to data")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # test python rewrite_data_text.py --noisy-path /mnt/data1/csj_enh_asr_simulated/noisy/eval1/noisy
    # --isolated-path /mnt/data1/csj_enh_asr_simulated/noisy/eval1/isolated --data-path data/eval1/
    args = get_args()

    rewrite_data_text = RewriteDataText(args.noisy_path, args.isolated_path, args.data_path)
    # rewrite
    rewrite_data_text.rewrite_segments()
    rewrite_data_text.rewrite_spk1()
    rewrite_data_text.rewrite_noise1()
    rewrite_data_text.rewrite_wav()
    rewrite_data_text.rewrite_text_spk2utt()
    rewrite_data_text.rewrite_text()
