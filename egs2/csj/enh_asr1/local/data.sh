#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=3
nj=-1
background_path=

simulated_data=/mnt/data1/csj_enh_asr_simulated/
help_message=$(cat << EOF
Usage: $0 [--stage <stage>] [--stop_stage <stop_stage>] [--nj <nj>]

  required argument:
    --background_path  : path to noise directory.    default="${background_path}"
    --simulated_data : path to simulated directory.  default="${simulated_data}"

    NOTE:
        This script is written in imitation of chime4's script.

  optional argument:
    [--stage]: 1 (default) or 2
    [--stop_stage]: 1 or 2 (default)
    [--nj]: number of parallel pool workers in Python
EOF
)


log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;
utils=./utils


if [ ! -e "${CSJDATATOP}" ]; then
    log "Fill the value of 'CSJDATATOP' of db.sh"
    exit 1
fi
if [ -z "${CSJVER}" ]; then
    log "Fill the value of 'CSJVER' of db.sh"
    exit 1
fi

if [ $# -ne 0 ] || [ -z "${background_path}" ]; then
    echo "${help_message}"
    exit 2
fi

if [ $# -ne 0 ] || [ -z "${simulated_data}" ]; then
    echo "${help_message}"
    exit 2
fi

train_set=train_nodup
train_dev=train_dev
recog_set="eval1 eval2 eval3"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Initial normalization of the data"
    local/csj_make_trans/csj_autorun.sh ${CSJDATATOP} data/csj-data ${CSJVER}
    local/csj_data_prep.sh data/csj-data

    for x in ${recog_set}; do
        local/csj_eval_data_prep.sh data/csj-data/eval ${x}
    done

    for x in train eval1 eval2 eval3; do
        local/csj_rm_tag_sp_space.sh data/${x}
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1 : Make a development set"
    utils/subset_data_dir.sh --first data/train 4000 data/${train_dev} # 6hr 31min
    n=$(($(wc -l < data/train/segments) - 4000))
    utils/subset_data_dir.sh --last data/train ${n} data/train_nodev

    # remove duplicated utterances in the training set
    utils/data/remove_dup_utts.sh 300 data/train_nodev data/${train_set} # 233hr 36min

fi

# eval1 eval2 eval3 train_dev train_nodev train_set を雑音重畳していく。

## added by kinouchi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2 : Simulate the noisy data for enhancement"

    if [ ! -e data/logs ]; then mkdir data/logs ; fi

    if [ -e ${simulated_data} ]; then rm -rf ${simulated_data} ; fi
    mkdir ${simulated_data}
    # backgroundを取ってきて、test(eval) とdev(train_dev) train(train_set)に分ける。
    # 雑音ノイズのファイルが少ないので、testは010の前半、devは010の後半を使う。
    if [ -e ${simulated_data}/background ]; then rm -rf ${simulated_data}/background ; fi
    mkdir ${simulated_data}/background
    mkdir ${simulated_data}/background/eval_dev ${simulated_data}/background/train ${simulated_data}/background/all
    cp ${background_path}/*_CAF.CH1.* ${simulated_data}/background/all                        # cafeのCH1のみを取得(DEBUG用)

    # 010をeval用に、その他はtrain用に使う
    mv ${simulated_data}/background/all/*_010_* ${simulated_data}/background/eval_dev
    mv ${simulated_data}/background/all/* ${simulated_data}/background/train
    rm -rf ${simulated_data}/background/all

    if [ -e ${simulated_data}/noisy ]; then rm -rf ${simulated_data}/noisy ; fi
    mkdir ${simulated_data}/noisy

    # test set
    echo "simulating now .... There are logs in data/logs/datasimulation_*_testset.log"
    loop=${recog_set}
    for x in ${loop}; do
        mkdir ${simulated_data}/noisy/${x}
        python3 local/CSJ_simulate_data_patched_parallel.py --nj ${nj} --scp_file data/${x}/wav.scp \
            --noise_path ${simulated_data}/background/eval_dev --use_data first --dist ${simulated_data}/noisy/${x}     > data/logs/datasimulation_${x}_testset.log
    done
    # valid set
    mkdir ${simulated_data}/noisy/${train_dev}
    python3 local/CSJ_simulate_data_patched_parallel.py --nj ${nj} --scp_file data/${train_dev}/wav.scp \
        --noise_path ${simulated_data}/background/eval_dev --use_data end --dist ${simulated_data}/noisy/${train_dev}   > data/logs/datasimulation_validset.log

    # train set
    mkdir ${simulated_data}/noisy/${train_set}
    python3 local/CSJ_simulate_data_patched_parallel.py --nj ${nj} --scp_file data/${train_set}/wav.scp \
        --noise_path ${simulated_data}/background/train --use_data all --dist ${simulated_data}/noisy/${train_set}      > data/logs/datasimulation_trainset.log
    log "finish to simlate noisy data"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3 : Rewrite Data path in data/"
    if [ ! -e data/logs ]; then mkdir data/logs ; fi
    if [ -e data_BK ]; then rm -rf data_BK ; fi
    cp -r data data_BK

    # test set
    for x in ${recog_set};do
        python3 local/rewrite_data_text.py --noisy-path ${simulated_data}/noisy/${x}/noisy --isolated-path ${simulated_data}/noisy/${x}/isolated --data-path data/${x} #> data/logs/rewrite_data_${x}_testset.log
        cat data/${x}/spk2utt | ${utils}/spk2utt_to_utt2spk.pl > data/${x}/utt2spk
    done

    # valid set
    python3 local/rewrite_data_text.py --noisy-path ${simulated_data}/noisy/${train_dev}/noisy --isolated-path ${simulated_data}/noisy/${train_dev}/isolated --data-path data/${train_dev} #> data/logs/rewrite_data_evalset.log
    cat data/${train_dev}/spk2utt | ${utils}/spk2utt_to_utt2spk.pl > data/${train_dev}/utt2spk

    # train set
    python3 local/rewrite_data_text.py --noisy-path ${simulated_data}/noisy/${train_set}/noisy --isolated-path ${simulated_data}/noisy/${train_set}/isolated --data-path data/${train_set} #> data/logs/rewrite_data_trainset.log
    cat data/${train_set}/spk2utt | ${utils}/spk2utt_to_utt2spk.pl > data/${train_set}/utt2spk
    
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
