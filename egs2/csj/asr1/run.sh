#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ~/tools/line_notificator.sh

train_set=train_nodup
valid_set=train_dev
#test_sets="dev_4k dev tedx-jp-10k "
# test_sets="eval1 eval2 eval3"
test_sets="train_dev"

asr_config=conf/tuning/train_asr_transformer_w2v2frontend.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"


line_notify "klab start"
# NOTE: The default settings require 4 GPUs with 32 GB memory
./asr.sh \
    --asr_args "--use_wandb true --wandb_project wav2vec2_csj_frontend" \
    --feats_normalize "" \
    --stage 11 \
    --ngpu 1 \
    --lang jp \
    --token_type char \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --gpu_inference true \
    --inference_config "${inference_config}" \
    --inference_asr_model "65epoch.pth" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/train_nodev/text" "$@" || line_notify "Klab failed"

line_notify "klab end"
