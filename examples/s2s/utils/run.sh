#!/bin/bash

# This script starts multiple instances of the Python script with different IDs,
# and redirects their outputs to corresponding log files.

# for id in {0..50}
# do
#    python /nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/utils/parquet_to_json2.py $id > "logs/${id}.log" 2>&1 &
# done

# wait
# echo "All processes are complete."


#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_id> <end_id>"
    exit 1
fi


for id in $(seq $1 $2)
do
   python /nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/utils/parquet_to_json2.py $id > "logs/${id}.log" 2>&1 &
done

wait
echo "All processes are complete."