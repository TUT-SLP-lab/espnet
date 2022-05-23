import logging
import glob
import numpy as np
import argparse
import os
import shutil
from joblib import Parallel, delayed
from scipy.io.wavfile import read, write

logging.basicConfig(level=logging.DEBUG)


class NoiseSuperimposition:
    def __init__(self, nj: int, scp_path: str, noise_path: str, use_data: str, dist: str):
        """
        nj : nuber jobs
        csj_path : path of CSJ
        noise_path : path to CHiME4 or CHiME3 or other noises
        """
        self.nj = nj
        self.scp_path = scp_path
        self.noise_path = noise_path
        self.use_data = use_data

        self.dist_noisy = dist + "/noisy"
        self.dist_isolated = dist + "/isolated"
        # make directory
        if os.path.exists(self.dist_noisy):
            shutil.rmtree(self.dist_noisy)
        if os.path.exists(self.dist_isolated):
            shutil.rmtree(self.dist_isolated)
        os.makedirs(self.dist_noisy)
        os.makedirs(self.dist_isolated)

        self.DEBUG = True

    def superimposition(self, csj_file: str, noise_file: str):
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
        if self.use_data == "first":
            print("use first")
            noise_data = noise_data[:half]
        if self.use_data == "end":
            print("use end")
            noise_data = noise_data[half:]

        # cleanデータのほうが長かったら、noiseを繰り返す。
        while len(clean_data) > len(noise_data):
            noise_data = np.concatenate([noise_data, noise_data])

        # モノラルなら無視(-R.wavと-L.wavがあるから)
        if clean_data.ndim == 2:
            return

        # 同じ長さにして重畳
        noise_data = noise_data[0 : len(clean_data)]
        noisy_data = clean_data + noise_data

        if clean_rate != noise_rate:
            logging.error("sampling rate is different")
            return

        # write noisy data
        noise_name = noise_file.split("_")[-1].split(".")  # [STR , CH0]
        noise_no = noise_file.split("_")[-2]  # 020
        csj_name = csj_file.split("/")[-1].split(".")[0]  # A01M0056
        # <dist_path>/A01M0056_STR.CH0.wav
        dist_file = f"{self.dist_noisy   }/{csj_name}-{noise_no}-{noise_name[0]}.{noise_name[1]}.wav"
        write(dist_file, noise_rate, noisy_data)

        # output clean file
        dist_file = f"{self.dist_isolated}/{csj_name}-{noise_no}-{noise_name[0]}.{noise_name[1]}.Clean.wav"
        write(dist_file, noise_rate, noisy_data)
        # output noise file
        dist_file = f"{self.dist_isolated}/{csj_name}-{noise_no}-{noise_name[0]}.{noise_name[1]}.Noise.wav"
        write(dist_file, noise_rate, noisy_data)
        return

    def run_parallel(self):

        # get wav path
        csj_path_list = []
        with open(self.scp_path) as scpf:
            for line in scpf:
                csj_path_list.append(line.split(" ")[2])
        noise_path_list = glob.glob(self.noise_path + "/*.wav", recursive=True)

        print(f"csj list: {csj_path_list}")
        print(f"noise list: {noise_path_list}")

        print("make simulation")

        # make core simulation
        for n in noise_path_list:
            Parallel(n_jobs=self.nj)([delayed(self.superimposition)(c, n) for c in csj_path_list])
            print(f"finished to simulate {n}")


def get_args():
    parser = argparse.ArgumentParser(description="Simulate data with parallel process")
    parser.add_argument(
        "--scp_file", type=str, required=True, help="path to scp_file",
    )
    parser.add_argument(
        "--noise_path", type=str, required=True, help="path to noise directory",
    )
    parser.add_argument(
        "--use_data",
        type=str,
        required=True,
        choices=["all", "first", "end"],
        default="all",
        help="where you will use data?, 'first', 'end', 'all",
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
    ns = NoiseSuperimposition(args.nj, args.scp_file, args.noise_path, args.use_data, args.dist)
    ns.run_parallel()
