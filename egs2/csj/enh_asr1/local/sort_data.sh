target_dir=$1

for file in ${target_dir}/* 
do 
    sort ${file} > tmp
    tmp > ${file} 
    rm tmp
done