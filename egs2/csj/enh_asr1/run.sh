#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup_multi_noisy
valid_set=train_dev_multi_noisy
test_sets="eval1 eval2 eval3"

enh_asr_config=conf/train_enh_asr_convtasnet_fbank_transformer.yaml
inference_config=conf/decode_asr_transformer.yaml
lm_config=conf/train_lm.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
# speed_perturb_factors="0.9 1.0 1.1"

use_word_lm=false
word_vocab_size=65000

background_path=/mnt/data1/database/CHiME4/data/audio/16kHz/backgrounds/
simulated_data=/mnt/data1/csj_enh_asr_simulated/

# NOTE: because dump is too big, dump and exp in mnt/ 
dumpdir=/mnt/data1/csj_enh_asr/dump
expdir=/mnt/data1/csj_enh_asr/exp

# NOTE: The default settings require 4 GPUs with 32 GB memory

./enh_asr.sh \
    --ngpu 4 \
    --stage 2 \
    --lang jp \
    --spk_num 1 \
    --ref_channel 3 \
    --local_data_opts "--background_path  ${background_path} --stage 5 --simulated_data ${simulated_data}"  \
    --nlsyms_txt data/nlsyms.txt \
    --token_type char \
    --feats_type raw \
    --feats_normalize utt_mvn \
    --enh_asr_config "${enh_asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --use_word_lm ${use_word_lm} \
    --word_vocab_size ${word_vocab_size} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --dumpdir  "${dumpdir}" \
    --expdir "${expdir}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"
    
    
#" data/local/other_text/text" "$@"

