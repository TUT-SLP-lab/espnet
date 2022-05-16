#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k
# Path to a directory containing extra annotations for CHiME4
# Run `local/data.sh` for more information.
extra_annotations=/mnt/data1/database/CHiME4/data/annotations

# train_set=tr05_simu_isolated_6ch_track
# valid_set=dt05_simu_isolated_6ch_track
# test_sets="et05_simu_isolated_6ch_track"

train_set=tr05_simu_isolated_1ch_track
valid_set=dt05_simu_isolated_1ch_track
test_sets="et05_simu_isolated_1ch_track"

asr_model=exp/asr_train_asr_transformer_raw_char_1gpu
lm_model=exp/lm_train_lm_char_sgd

inference_asr_model=valid.acc.ave_10best.pth
inference_lm_model=37epoch.pth

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 4 \
    --spk_num 1 \
    --ref_channel 1 \
    --local_data_opts "--extra-annotations ${extra_annotations} --stage 1 --stop-stage 2" \
    --enh_config conf/tuning/train_enh_conv_tasnet.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    --score_with_asr true \
    --asr_exp  "${asr_model}" \
    --lm_exp  "${lm_model}" \
    --inference_asr_model "${inference_asr_model}" \
    --inference_lm "${inference_lm_model}" \
    --gpu_inference true \
    --inference_nj 1 \
    "$@"
