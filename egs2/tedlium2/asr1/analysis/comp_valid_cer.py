import matplotlib.pyplot as plt
import numpy as np
import sys
import re 

log_path = sys.argv[1]
cer = []
loss = []

with open(log_path, "r") as f:
    line = f.readline()
    
    while line:
        if re.search("results", line):
            match_cer = re.search(r'.*cer_interctc_layer19=(\d+\.\d+),.*cer_interctc_layer20=(\d+\.\d+)', line)
            cer.append([match_cer.group(1), match_cer.group(2)])

            match_loss = re.search(r'.*loss_interctc_layer19=(\d+\.\d+),.*loss_interctc_layer20=(\d+\.\d+)', line)
            loss.append([match_loss.group(1), match_loss.group(2)])

        line = f.readline()

cer = np.array(cer, dtype=float)
plt.title("cer vs epoch")
plt.xlabel("epoch")
plt.ylabel("cer")
plt.plot(cer[:, 0], label="lower")
plt.plot(cer[:, 1], label="upper")
plt.ylim(0, 0.5)
plt.grid()
plt.legend()
plt.savefig("analysis/cer_phoneme.png")
plt.clf()

loss = np.array(loss, dtype=float)
plt.title("loss vs epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(loss[:, 0], label="lower")
plt.plot(loss[:, 1], label="upper")
plt.grid()
plt.legend()
plt.savefig("analysis/loss_phoneme.png")
