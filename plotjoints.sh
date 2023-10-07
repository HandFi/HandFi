folder_path="./runsijie5/exp2"

if [ ! -d "$folder_path" ]; then
    mkdir -p "$folder_path"
fi

for i in $(seq 0 1 10) 
do
    python plotjoints.py --k 0 --j $i --folder runsijie5 --exp exp2
done