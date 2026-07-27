import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
import math

model_name = "asr_att_ctc_for_plot"
model_dir = f"../exp/{model_name}"
output_dir = "images/attention_map"
head_id = 4

att_w1 = torch.load(f"{model_dir}/att_ws/AlGore_2009-0001304-0002346/encoder.multihead_attn.50ep.pth")
fig, ax = plt.subplots()
att_w1 = att_w1[head_id]
sns.heatmap(att_w1[::-1])
ax.set_xlabel("Frame", fontsize=18)
ax.set_ylabel("Layer", fontsize=18)
ax.set_xticks([x for x in range(att_w1.shape[1]) if x % 40 == 0])
ax.set_xticklabels([x for x in range(att_w1.shape[1]) if x % 40 == 0], rotation=0, fontsize=16)
# ax.set_yticklabels([x for x in range(1, 10, 1)], rotation=0, fontsize=12)
ax.set_yticklabels([x for x in range(9, 0, -1)], rotation=0, fontsize=16)
p1 = f"{output_dir}/{model_name}_low.png"
fig.tight_layout()
plt.savefig(p1, bbox_inches="tight")

plt.close(fig)
att_w2 = torch.load(f"{model_dir}/att_ws/AlGore_2009-0001304-0002346/encoder.multihead_attn2.50ep.pth")
fig, ax = plt.subplots()
att_w2 = att_w2[head_id]
sns.heatmap(att_w2[::-1])
ax.set_xlabel("Frame", fontsize=18)
ax.set_ylabel("Layer", fontsize=18)
ax.set_xticks([x for x in range(att_w2.shape[1]) if x % 40 == 0])
ax.set_xticklabels([x for x in range(att_w2.shape[1]) if x % 40 == 0], rotation=0, fontsize=16)
# ax.set_yticklabels([x for x in range(10, 19, 1)], rotation=0, fontsize=12)
ax.set_yticklabels([x for x in range(18, 9, -1)], rotation=0, fontsize=16)
p2 = f"{output_dir}/{model_name}_upp.png"
fig.tight_layout()
plt.savefig(p2, bbox_inches="tight")
