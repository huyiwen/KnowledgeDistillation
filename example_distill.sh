#!/bin/bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0,1

case "$1" in
    -d|--daemon)
        $0 < /dev/null &> /dev/null & disown
        exit 0
        ;;
    *)
        ;;
esac


BARK_TOKEN=$(cat BARK_TOKEN)

# post [Completed/Failed] + [Method]
function post() {
if [ $1 -ne 0 ]; then
	curl -X POST "https://api.day.app/$BARK_TOKEN/" \
		-H 'Content-Type: application/x-www-form-urlencoded; charset=utf-8' \
		-d "title=Failed+$2&body=$(LANG=en_us_88591; date +"[%Y.%m.%d %H:%M:%S]")+$(tail -2 $log_path)&group=Failed&icon=https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png&level=timeSensitive"
	echo ""
	echo -e "\033[1;31m    Failed:\033[0m $2"
    echo "$(tail -20 $log_path)"
    echo ""
else
	curl -X POST "https://api.day.app/$BARK_TOKEN/" \
		-H 'Content-Type: application/x-www-form-urlencoded; charset=utf-8' \
		-d "title=Completed+$2&body=$(LANG=en_us_88591; date +"[%Y.%m.%d %H:%M:%S]")+$(tail -15 $log_path | grep cc | tail -1)&group=Success&icon=https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png&level=passive"
	echo ""
	echo -e "\033[1;31m   Success:\033[0m $2"
    echo "$(tail -20 $log_path)"
    echo ""
fi
}

function _train() {
    nohup python3 -u distill.py $@ >> $log_path 2>&1
    post $? $model_tag
}

function train() {

    model_tag="$1"
    shift
    log_path=logs/$model_tag$(LANG=en_us_88591; date +"%Y.%m.%d_%H:%M:%S.log")
    echo -e "\033[1;31m Model Tag:\033[0m $model_tag"
    echo -e "\033[1;31m  Log Path:\033[0m $log_path"
    echo -e "\033[1;31m Arguments:\033[0m $@"
    _train $@ &
    pid=$!
    echo -e "\033[1;31m   Process:\033[0m $pid"
    sleep 0.1
    while [[ $(jobs -r) ]]; do
        echo -en "\033[2K\r     $(tail -n 1 $log_path | cut -c1-$(($COLUMNS-10)))"
        sleep 0.1
    done

}

# Teacher 77.85%
# train Teacher --train-teacher 1 --train-student 0

# KDBaseline 68.05%
# train baseline


###########     EMB      ###########

# EMB 73.99%
# train EMB --use-mpo --mpo-type "embedding" --embedding-input-shape 19 4 2 7 20 --embedding-output-shape 10 3 1 1 10

# EMB 73.60%
# train EMB --use-mpo --mpo-type "embedding" --embedding-input-shape 19 4 4 7 10 --embedding-output-shape 10 3 1 1 10

# EMB
train EMB --use-mpo --mpo-type "embedding" --embedding-input-shape 38 4 1 2 70 --embedding-output-shape 30 1 1 1 10


###########      FC      ###########

# FC 68.88%
# train FC --use-mpo --mpo-type "fc" --fc1-input-shape 10 2 3 10 --fc1-output-shape 6 2 2 8 --fc2-input-shape 6 2 2 8 --fc2-output-shape 5 1 1 1

# FC 69.16% simply add 1s in the middle
# train FC --use-mpo --mpo-type "fc" --fc1-input-shape 10 2 1 3 10 --fc1-output-shape 6 2 1 2 8 --fc2-input-shape 6 2 1 2 8 --fc2-output-shape 5 1 1 1 1


###########     LSTM     ###########

# LSTM 31.29%
# train LSTM --use-mpo --mpo-type "lstm" --xh-input-shape 10 3 1 2 5 --xh-output-shape 10 3 2 4 5 --hh-input-shape 10 3 1 2 5 --hh-output-shape 10 3 2 4 5

# LSTM 52.57%
# train LSTM --use-mpo --mpo-type "lstm" --xh-input-shape 10 3 1 1 10 --xh-output-shape 10 3 1 4 10 --hh-input-shape 10 3 1 1 10 --hh-output-shape 10 3 1 4 10

# LSTM 64.57%
# train LSTM --use-mpo --mpo-type "lstm" --xh-input-shape 20 15 --xh-output-shape 40 30 --hh-input-shape 20 15 --hh-output-shape 40 30

# LSTM 64.93%
# train LSTM --use-mpo --mpo-type "lstm" --xh-input-shape 300 1 --xh-output-shape 1200 1 --hh-input-shape 300 1 --hh-output-shape 1200 1

# LSTM 64.41%
# train LSTM --use-mpo --mpo-type "lstm" --xh-input-shape 30 10 --xh-output-shape 30 40 --hh-input-shape 30 10 --hh-output-shape 30 40

# LSTM
train LSTM --use-mpo --mpo-type "lstm" --xh-input-shape 30 1 1 10 --xh-output-shape 30 1 1 40 --hh-input-shape 30 1 1 10 --hh-output-shape 30 1 1 40

# LSTM
train LSTM --use-mpo --mpo-type "lstm" --xh-input-shape 300 1 --xh-output-shape 1 1200 --hh-input-shape 300 1 --hh-output-shape 1 1200

# LSTM
train LSTM --use-mpo --mpo-type "lstm" --xh-input-shape 300 1 1 --xh-output-shape 1200 1 1 --hh-input-shape 300 1 1 --hh-output-shape 1200 1 1

# LSTM
train LSTM --use-mpo --mpo-type "lstm" --xh-input-shape 300 1 1 --xh-output-shape 1 1 1200 --hh-input-shape 1 1 300 --hh-output-shape 1 1 1200


###########   LSTM_FC    ###########

# LSTM_FC 57.40%
# train LSTM_FC --use-mpo --mpo-type "lstm" "fc" --fc1-input-shape 10 2 3 10 --fc1-output-shape 6 2 2 8 --fc2-input-shape 6 2 2 8 --fc2-output-shape 5 1 1 1 --xh-input-shape 10 3 1 10 --xh-output-shape 10 3 4 10 --hh-input-shape 10 3 1 10 --hh-output-shape 10 3 4 10


########### EMB_LSTM_FC  ###########

# EMB_LSTM_FC 66.66%
# train EMB_LSTM_FC --use-mpo --mpo-type "lstm" "fc" "embedding" --fc1-input-shape 10 2 3 10 --fc1-output-shape 6 2 2 8 --fc2-input-shape 6 2 2 8 --fc2-output-shape 5 1 1 1 --xh-input-shape 10 3 1 10 --xh-output-shape 10 3 4 10 --hh-input-shape 10 3 1 10 --hh-output-shape 10 3 4 10 --embedding-input-shape 19 8 7 20 --embedding-output-shape 10 3 1 10

# EMB_LSTM_FC 65.68%
# train EMB_LSTM_FC --use-mpo --mpo-type "lstm" "fc" "embedding" --fc1-input-shape 10 2 1 3 10 --fc1-output-shape 6 2 1 2 8 --fc2-input-shape 6 2 1 2 8 --fc2-output-shape 5 1 1 1 1 --xh-input-shape 10 3 1 1 10 --xh-output-shape 10 3 2 2 10 --hh-input-shape 10 3 1 1 10 --hh-output-shape 10 3 2 2 10 --embedding-input-shape 19 4 2 7 20 --embedding-output-shape 10 3 1 1 10

