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
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.plot(dev, label="dev", color="black", marker="o", markersize=8, linewidth=3)
    plt.plot(test, label="test", color="black", marker="s", markersize=8, linewidth=3)
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.xticks(range(len(dev)), xticks, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(frameon=False, ncol=2, fontsize=15)
    plt.savefig(output_path+".pdf")
    plt.savefig(output_path+".png")
    plt.clf()

def main():
    div_path = sys.argv[1]
    div_out = f"images/layer/{os.path.splitext(os.path.basename(div_path))[0]}"
    plot(div_path, "Number of lower layers", "WER (%)", range(3, 16, 3), div_out)

    attn_path = sys.argv[2]
    attn_out = f"images/layer/{os.path.splitext(os.path.basename(attn_path))[0]}"
    plot(attn_path, "Number of attention module", "WER (%)", range(1, 5), attn_out)


if __name__ == "__main__":
    main()