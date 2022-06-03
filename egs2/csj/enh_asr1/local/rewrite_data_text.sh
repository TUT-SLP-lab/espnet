# bash local/rewrite_data_text.sh [noisy-path] [clean_path] [isolated-path] [data-path] 
 
# bash local/rewrite_data_text.sh /mnt/data1/csj_enh_asr_simulated/noisy/eval1/noisy /mnt/data1/csj_enh_asr_simulated/splited_wav/eval1 /mnt/data1/csj_enh_asr_simulated/noisy/eval1/isolated data_tmp/eval1/

noisy_path=$1
clean_path=$2
isolated_path=$3
data_path=$4
data_clean_path=$5

utils=utils

if [ -e $data_path ]; then rm -rf $data_path ; fi
mkdir $data_path

# rewrite spk1
cp ${isolated_path}/spk1.scp ${data_path}/spk1.scp
echo "finish to rewrite spk1"

# rewrite noise1
cp ${isolated_path}/noise1.scp ${data_path}/noise1.scp
echo "finish to rewrite noise"

# rewrite wav.scp
cp ${noisy_path}/wav.scp ${data_path}/wav.scp # 新規上書き
echo "finish to rewrite wav"

# rewrite utt2spk
cat ${data_path}/wav.scp | sed -r 's/^(.*) .*\/([^_]*)_.*/\1 \2/' > ${data_path}/utt2spk 
cat ${data_path}/utt2spk | ${utils}/utt2spk_to_spk2utt.pl > ${data_path}/spk2utt

echo "finish to rewrite utt2spk and spk2utt"


# rewrite text
# もとのtextデータからlineを取ってきて、それをutt2spkからサーチして増幅

cat ${data_clean_path}/text | while read line
do
    utt=`echo ${line%% *}` #  A01M0141_100000_200000
    grep ${utt} ${data_path}/utt2spk  | while read line_u2s
    do
        utt_u2s=`echo ${line_u2s%% *}` 
        echo ${line/${utt}/${utt_u2s}} >> tmp
    done
done

sort tmp > ${data_path}/text # 新規上書き

rm tmp
echo "finish to rewrite text"