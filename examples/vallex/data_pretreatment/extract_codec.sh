GPU_NUM=8
NUM_P=16
save_home="/home/develop_dataset/codec"
tsv="/home/develop_dataset/info_bilibili.tsv"
GPU_IDX=0
mkdir -p ${save_home}
for ((i=0; i<NUM_P; i++));
do  
    temp_file=$ori_home/${i}_codec0.tsv
    echo run $i $GPU_IDX
    DEVICEIDX=$[${GPU_IDX}%(${GPU_NUM})]
    CUDA_VISIBLE_DEVICES=$DEVICEIDX \
    python -u extract_codec.py \
        --tsv  ${tsv} \
        --save-home ${save_home} \
        --pro-idx ${i} \
        --pro-total ${NUM_P} &
    GPU_IDX=$[$GPU_IDX+1]
done
wait



