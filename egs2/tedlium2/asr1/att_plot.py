import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
import math


att_w1 = torch.load("./exp/asr_test_prop/att_ws/AlGore_2009-0001304-0002346/encoder.multihead_attn.50ep.pth")
fig, ax = plt.subplots()
att_w1 = att_w1[3]
sns.heatmap(att_w1)
ax.set(xlabel ='frame',ylabel='Layer' )
ax.set_xticks([x for x in range(att_w1.shape[1]) if x % 20 == 0])
ax.set_xticklabels([x for x in range(att_w1.shape[1]) if x % 20 == 0])
# ax.set_yticks([x*0.1 for x in range(0, 95, 5)])
# ax.set_yticklabels([math.ceil(x*0.1) if x % 10 != 0 else "" for x in range(0, 95, 5)])
# ax.set_yticks([x for x in range(1, 10)])
ax.set_yticklabels([x for x in range(1, 10)])
p1 = "./attn1-1.png"
plt.savefig(p1)
# print([math.ceil(x*0.1) if x % 10 != 0 else "" for x in range(0, 95, 5)])
# ax.set_yticks([x for x in range(1, 10)], top=True)
# ax.set_yticklabels([x for x in range(1, 10)])

plt.clf()
att_w2 = torch.load("./exp/asr_test_prop/att_ws/AlGore_2009-0001304-0002346/encoder.multihead_attn2.50ep.pth")
fig, ax = plt.subplots()
att_w2 = att_w2[4]
sns.heatmap(att_w2)
ax.set(xlabel ='frame',ylabel='Layer' )
ax.set_xticks([x for x in range(att_w2.shape[1]) if x % 20 == 0])
ax.set_xticklabels([x for x in range(att_w2.shape[1]) if x % 20 == 0])
# ax.set_yticks([x*0.1 for x in range(0, 95, 5)])
# ax.set_yticklabels([math.ceil(x*0.1) if x % 10 != 0 else "" for x in range(90, 185, 5)])
ax.set_yticklabels([x for x in range(10, 19)])
p2 = "./attn2-1.png"
plt.savefig(p2)

# plt.xlabel("frame", fontsize=10)
# plt.ylabel("Layer", fontsize=10)

# w, h = plt.figaspect(1.0 / len(att_w))
# fig = plt.Figure(figsize=(w * 1.3, h * 1.3))
# axes = fig.subplots(111)
# att_w = att_w[0]
# if len(att_w) == 1:
#     axes = [axes]

# for ax, aw in zip(axes, att_w):
#     ax.imshow(aw.astype(np.float32), aspect="auto")
#     ax.set_title(f"frame-wise attention weight")
#     ax.set_xlabel("Input")
#     ax.set_ylabel("Output")
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# fig, ax = plt.subplots()
# im = ax.imshow(att_w[4], aspect=20)
# plt.colorbar(im)
# sns.heatmap(att_w1[4])

# p1 = "./attn2.png"
# plt.savefig(p)

