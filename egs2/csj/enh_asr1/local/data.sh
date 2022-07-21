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
stop_stage=5
nj=-1
background_path=

simulated_data=
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

# segmentから一度utils/split_scp.pl を使って切り出す。
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: split CSJ by segment file"
    loop=${recog_set}" ${train_dev} ${train_set}"
    if [ -e ${simulated_data}/splited_wav ]; then rm -rf ${simulated_data}/splited_wav ; fi
    mkdir ${simulated_data}/splited_wav
    
    for x in ${loop}; do
        python pyscripts/audio/format_wav_scp.py data/${x}/wav.scp ${simulated_data}/splited_wav/${x} --segments data/${x}/segments --audio-format wav
        cat ${simulated_data}/splited_wav/${x}/wav.scp > data/${x}/wav.scp
        rm data/${x}/segments
    done
fi

# eval1 eval2 eval3 train_dev train_nodev train_set を雑音重畳していく。

## added by kinouchi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3 : Simulate the noisy data for enhancement"

    if [ ! -e data/logs ]; then mkdir data/logs ; fi

    # backgroundを取ってきて、test(eval) とdev(train_dev) train(train_set)に分ける。
    # 雑音ノイズのファイルが少ないので、testは010の前半、devは010の後半を使う。
    if [ -e ${simulated_data}/background ]; then rm -rf ${simulated_data}/background ; fi
    mkdir ${simulated_data}/background

    mkdir ${simulated_data}/background/eval_dev ${simulated_data}/background/train ${simulated_data}/background/all
    cp ${background_path}/* ${simulated_data}/background/all   # *すべてを取得

    # 010をeval用に、それ以外はtrain用に使う
    mv ${simulated_data}/background/all/*_010_* ${simulated_data}/background/eval_dev
    mv ${simulated_data}/background/all/* ${simulated_data}/background/train
    rm -rf ${simulated_data}/background/all

    if [ -e ${simulated_data}/noisy ]; then rm -rf ${simulated_data}/noisy ; fi
    mkdir ${simulated_data}/noisy

    # test set
    log "simulating now .... There are logs in data/logs/datasimulation_*_testset.log"
    loop=${recog_set}
    for x in ${loop}; do
        mkdir ${simulated_data}/noisy/${x}
        python3 local/CSJ_simulate_data_patched_parallel.py --nj ${nj} --splited_wav_dir ${simulated_data}/splited_wav/${x} \
            --noise_path ${simulated_data}/background/eval_dev --data_type test --dist ${simulated_data}/noisy/${x}    #> data/logs/datasimulation_${x}.log
    done
    log "finished to simualte ${x} noisy set"
    
    # valid set
    mkdir ${simulated_data}/noisy/${train_dev}
    python3 local/CSJ_simulate_data_patched_parallel.py --nj ${nj} --splited_wav_dir ${simulated_data}/splited_wav/${train_dev} \
        --noise_path ${simulated_data}/background/eval_dev --data_type valid --dist ${simulated_data}/noisy/${train_dev} #> data/logs/datasimulation_validset.log
    log "finished to simualte valid noisy set"

    # train set
    mkdir ${simulated_data}/noisy/${train_set}
    python3 local/CSJ_simulate_data_patched_parallel.py --nj ${nj} --splited_wav_dir ${simulated_data}/splited_wav/${train_set} \
        --noise_path ${simulated_data}/background/train --data_type train --dist ${simulated_data}/noisy/${train_set}    #> data/logs/datasimulation_trainset.log
    log "finish to simlate train noisy data"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4 : make simulated kaldi file in data/"
    if [ ! -e data/logs ]; then mkdir data/logs ; fi

    loop=${recog_set}" ${train_dev} ${train_set}"
    for x in ${loop};do
        local/rewrite_data_text.sh ${simulated_data}/noisy/${x}/noisy \
            ${simulated_data}/splited_wav/${x} \
            ${simulated_data}/noisy/${x}/isolated \
            data/${x}_simulated/ data/${x}

        ${utils}/sort_data.sh data/${x}_simulated/
        log "${x} will processed"
    done
fi



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5 : conbine real and simulated data"
    loop="${train_dev} ${train_set}" # avoid test set
    for x in ${loop};do
        cp data/${x}/wav.scp data/${x}/spk1.scp
        log "combine ${x} and ${x}_simulated"
        <data/${x}_simulated/wav.scp awk '{print($1, "SIMU")}' > data/${x}_simulated/utt2category
        <data/${x}/wav.scp awk '{print($1, "CLEAN")}' > data/${x}/utt2category

        utils/combine_data.sh --extra_files "utt2category spk1.scp" \
            data/${x}_multi_noisy data/${x}_simulated data/${x} 
    done

fi


log "Successfully finished. [elapsed=${SECONDS}s]"
