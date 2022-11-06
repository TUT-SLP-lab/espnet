#!/bin/sh

weights=(0.1 0.3 0.5 0.7 0.9 1.1)
for ms in ${weights[@]}
do
    for aps in ${weights[@]}
    do
        for sps in ${weights[@]}
        do
            ./run.sh \
            --stage 12 --stop-stage 13 \
            --asr_exp "exp/asr_asr_transformer_aps" \
            --asr_exp_add "exp/asr_transformer_sps" \
            --lm_exp "exp/lm_mainiti" \
            --lm_exp_sub "exp/lm_aps" \
            --lm_exp_sub_2 "exp/lm_sps" \
            --inference_tag "proposed/ms"$ms"_aps"$aps"_sps"$sps"" \
            --inference_args "--lm_weight $ms --lm_add_weight $aps --lm_add_2_weight $sps"
        done
    done
done


# ./run.sh \
#     --stage 11 --stop-stage 12 \
#     --asr_exp "exp/asr_rnn_aps" \
#     --asr_exp_add "exp/asr_rnn_sps" \
#     --lm_exp "exp/lm_mainiti" \
#     --lm_exp_sub "exp/lm_aps" \
#     --lm_exp_sub_2 "exp/lm_sps" \
#     --inference_tag "test1-5" \
#     --inference_args "--lm_weight 1.1 --lm_add_weight 0.5 --lm_add_2_weight 0.3"

# ./run.sh \
# --stage 12 --stop-stage 12 \
# --asr_exp "exp/asr_rnn_aps" \
# --asr_exp_add "exp/asr_rnn_sps" \
# --lm_exp "exp/lm_mainiti" \
# --lm_exp_sub "exp/lm_aps" \
# --lm_exp_sub_2 "exp/lm_sps" 

#./run.sh --stage 11 --stop-stage 12 --asr_exp "exp/asr_rnn_aps" --asr_exp_add "exp/asr_rnn_sps" --lm_exp "exp/lm_mainiti" --lm_exp_sub "exp/lm_aps" --lm_exp_sub_2 "exp/lm_sps" --inference_tag "dra" --inference_args "--lm_weight 0.9 --lm_add_weight 0.9 --lm_add_2_weight 0.0"
#./run.sh --stage 11 --stop-stage 12 --asr_exp "exp/asr_rnn_aps" --asr_exp_add "exp/asr_rnn_sps" --lm_exp "exp/lm_mainiti" --lm_exp_sub "exp/lm_aps" --lm_exp_sub_2 "exp/lm_sps" 

#&& send-slack-msg "finish" || send-slack-msg "failure"