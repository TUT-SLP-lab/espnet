#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

project_name="labor_xlsr_ctc_asr"

. ~/tools/line_notificator.sh

train_set=train_nodup
valid_set=train_dev
test_sets="dev eval1 eval2 eval3" # tedx-jp-10k"

asr_config=conf/tuning/train_asr_ctc_w2v_large_finetuning.yaml
inference_config=conf/decode_w2v_ctc_asr.yaml
lm_config=conf/train_lm.yaml

add_lm_exp=exp/laboro_lm_train_lm_jp_char
sub_lm_exp=exp/csj_lm_train_lm_jp_char

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

add_lm_weight_list=(0.8)
sub_lm_weight_list=(0.35)

line_notify "start $project_name"
for sub_lm_weight in "${sub_lm_weight_list[@]}"; do
	for add_lm_weight in "${add_lm_weight_list[@]}"; do
		echo "add_lm_weight: $add_lm_weight  sub_lm_weight: $sub_lm_weight"
		# NOTE: The default settings require 4 GPUs with 32 GB memory
		./asr.sh \
			--asr_args "--use_wandb true --wandb_project $project_name" \
			--feats_normalize "" \
			--ngpu 4 \
			--stage 11 \
			--lang jp \
			--dumpdir "/mnt/WDB_8TSSD/csj_dump" \
			--token_type char \
			--feats_type raw \
			--use_lm true \
			--asr_config "${asr_config}" \
			--inference_config "${inference_config}" \
			--inference_args "--add_lm_weight $add_lm_weight --sub_lm_weight $sub_lm_weight" \
			--inference_tag "decode_w2v_ctc_asr_lagest_add_w${add_lm_weight}_sub_w${sub_lm_weight}" \
			--inference_asr_model "latest.pth" \
			--gpu_inference true \
			--inference_nj 512 \
			--lm_config "${lm_config}" \
			--train_set "${train_set}" \
			--valid_set "${valid_set}" \
			--test_sets "${test_sets}" \
			--add_lm_exp "${add_lm_exp}" \
			--sub_lm_exp "${sub_lm_exp}" \
			--speed_perturb_factors "${speed_perturb_factors}" \
			--lm_train_text "data/train_nodev/text" "$@" || line_notify "failue $project_name"
	done
done

line_notify "end $project_name"
