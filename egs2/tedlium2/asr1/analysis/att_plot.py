import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
import math


att_w1 = torch.load("../exp/asr_decode_step-by-step_with_phn/att_ws/AlGore_2009-0001304-0002346/encoder.multihead_attn.50ep.pth")
fig, ax = plt.subplots()
att_w1 = att_w1[4]
sns.heatmap(att_w1)
ax.set(xlabel ='frame',ylabel='Layer' )
ax.set_xticks([x for x in range(att_w1.shape[1]) if x % 20 == 0])
ax.set_xticklabels([x for x in range(att_w1.shape[1]) if x % 20 == 0], rotation=0)
# ax.set_yticklabels([x for x in range(9, 0, -1)], rotation=0)
ax.set_yticklabels([x for x in range(1, 10, 1)], rotation=0)
p1 = "images/decode_step-by-step_with_phn_lower.png"
plt.savefig(p1)

plt.clf()
att_w2 = torch.load("../exp/asr_decode_step-by-step_with_phn/att_ws/AlGore_2009-0001304-0002346/encoder.multihead_attn2.50ep.pth")
fig, ax = plt.subplots()
att_w2 = att_w2[4]
sns.heatmap(att_w2)
ax.set(xlabel ='frame',ylabel='Layer' )
ax.set_xticks([x for x in range(att_w2.shape[1]) if x % 20 == 0])
ax.set_xticklabels([x for x in range(att_w2.shape[1]) if x % 20 == 0], rotation=0)
# ax.set_yticklabels([x for x in range(18, 9, -1)], rotation=0)
ax.set_yticklabels([x for x in range(10, 19, 1)], rotation=0)
p2 = "images/decode_step-by-step_with_phn_upper.png"
plt.savefig(p2)
# print(att_w2.shape)

# plt.xlabel("frame", fontsize=10)
# plt.ylabel("Layer", fontsize=10)

