folder_path="./runsijie5/exp1"

if [ ! -d "$folder_path" ]; then
    mkdir -p "$folder_path"
fi

for i in $(seq 0 1 15) 
do
    python plotmask.py --k 0 --j $i --folder runsijie5 --exp exp1
done