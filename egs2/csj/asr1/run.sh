#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="eval1 eval2 eval3"


#ここでは音声認識モデルの種類を選べる
#今回はTransformerモデル
#他には、train_asr_rnn.yamlとすると、RNNモデル
#train_asr_conformer.yamlとするとConformerモデル
asr_config=conf/train_asr_transformer.yaml 

#ここでは音声認識モデルを用いた際のデコード設定ファイルを指定
#Shallow Fusionの言語モデル重みを変えたりできる
inference_config=conf/decode_asr.yaml 

#言語モデルについての設定ファイル       
lm_config=conf/train_lm.yaml 

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

# NOTE: The default settings require 4 GPUs with 32 GB memory
./asr.sh \
    --ngpu 4 \
    --lang jp \
    --token_type char \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/train_nodev/text" "$@"
