# bash local/rewrite_data_text.sh [noisy-path] [clean_path] [isolated-path] [data-path] 
 
# bash local/rewrite_data_text.sh /mnt/data1/csj_enh_asr_simulated/noisy/eval1/noisy /mnt/data1/csj_enh_asr_simulated/splited_wav/eval1 /mnt/data1/csj_enh_asr_simulated/noisy/eval1/isolated data_tmp/eval1/

noisy_path=$1
clean_path=$2
isolated_path=$3
data_path=$4

utils=utils

# rewrite spk1
for file in ${isolated_path}/*.Clean.wav
do
    wav_name=`echo ${file##*/}` && wav_name=`echo ${wav_name%%.*}` && echo "${wav_name} cat ${file} | " >> tmp &
done
wait

sort tmp > "${data_path}/spk1.scp" # 新規上書き
rm tmp
echo "finish to rewrite spk1"

# rewrite noise1
for file in ${isolated_path}/*.Noise.wav
do
    wav_name=`echo ${file##*/}` && wav_name=`echo ${wav_name%%.*}` && echo "${wav_name} cat ${file} |" >> tmp &
done
wait

sort tmp > "${data_path}/noise1.scp" # 新規上書き
rm tmp
echo "finish to rewrite noise"

# rewrite wav
## noisy data

for file in ${noisy_path}/${spk}*.wav
do
    wav_name=`echo ${file##*/}` && \
    wav_name=`echo ${wav_name%%.*}` && \
    echo "${wav_name} ${file}  " >> tmp && \
    utt=$wav_name && \
    spk=`echo ${utt%%_*}` && \
    echo "${utt} ${spk}" >> tmp_utt2spk &
done

wait

## clean data
cat ${clean_path}/wav.scp >> tmp
cat ${data_path}/utt2spk >> tmp_utt2spk

sort tmp > "${data_path}/wav.scp" # 新規上書き
sort tmp_utt2spk > "${data_path}/utt2spk" 
cat ${data_path}/utt2spk | ${utils}/utt2spk_to_spk2utt.pl > ${data_path}/spk2utt

rm tmp tmp_utt2spk
echo "finish to rewrite wav and utt2spk spk2utt"



# rewrite text
# もとのtextデータからlineを取ってきて、それをutt2spkからサーチして増幅

cat ${data_path}/text | while read line
do
    utt=`echo ${line%% *}` #  A01M0141_100000_200000
    grep $utt ${data_path}/utt2spk | while read line_u2s
    do
        utt_u2s=`echo ${line_u2s%% *}` && echo ${line/${utt}/${utt_u2s}} >> tmp &
    done
    wait 
done

sort tmp > "${data_path}/text" # 新規上書き

rm tmp
echo "finish to rewrite text"

# remove segment
rm ${data_path}/segments