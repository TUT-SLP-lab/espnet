import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def plot(log_path, xlabel, ylabel, xticks, output_path):
    dev = []
    test = []

    with open(log_path, "r") as f:
        line = f.readline()
        
        while line:
            dev.append(float(line.split(",")[1]))
            test.append(float(line.split(",")[2]))

            line = f.readline()
        
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(dev, label="dev", color="black", marker="o", markersize=4)
    plt.plot(test, label="test", color="black", marker="s", markersize=4)
    plt.xticks(range(len(dev)), xticks)
    plt.legend(frameon=False, ncol=2)
    plt.savefig(output_path)
    plt.clf()

def main():
    div_path = sys.argv[1]
    div_out = f"images/layer/{os.path.splitext(os.path.basename(div_path))[0]}"
    plot(div_path, "Divide position", "WER (%)", range(3, 16, 3), div_out)

    attn_path = sys.argv[2]
    attn_out = f"images/layer/{os.path.splitext(os.path.basename(attn_path))[0]}"
    plot(attn_path, "Number of attention module", "WER (%)", range(1, 4), attn_out)


if __name__ == "__main__":
    main()