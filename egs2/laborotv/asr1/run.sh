#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#project_name="research_laboro_conformer_lower_bound"
project_name="research_laboro_lower_bound_finetuned_csj"
train_set=train_nodev
valid_set=dev_4k
test_sets="eval1 eval2 eval3 dev_4k dev tedx-jp-10k"

#asr_config=conf/train_asr_conformer.yaml
asr_config=conf/tuning/train_asr_transformer3_w2v_large_lv60_960h_finetuning_last_1layer.yaml

inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

# NOTE: The default settings require 4 GPUs with 32 GB memory
./asr.sh \
    --asr_args "--use_wandb true --wandb_project $project_name" \
    --ngpu 4 \
    --nj 64 \
    --stage 6 \
    --stop_stage 8 \
    --inference_nj 64 \
    --lang jp \
    --token_type char \
    --feats_type raw \
    --use_lm true \
    --dumpdir /mnt/WDB_8TSSD/laboro_dump \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "valid.acc.best.pth"\
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/${train_set}/text" "$@"
