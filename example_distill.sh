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
	echo "Failed $2 $(tail -20 $log_path)"
else
	curl -X POST "https://api.day.app/$BARK_TOKEN/" \
		-H 'Content-Type: application/x-www-form-urlencoded; charset=utf-8' \
		-d "title=Completed+$2&body=$(LANG=en_us_88591; date +"[%Y.%m.%d %H:%M:%S]")+$(tail -5 $log_path | grep Prec)&group=Success&icon=https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png&level=passive"
	echo ""
	echo "Success $2 $(tail -20 $log_path)"
fi
}

function train() {

    first_arg="$1"
    shift
    log_path=logs/$(LANG=en_us_88591; date +"%Y.%m.%d_%H:%M:%S.log")
    echo " Model Tag: $first_arg"
    echo "  Log Path: $log_path"
    echo " Arguments: $@"
    nohup python3 -u distill.py $@ >> $log_path 2>&1
    post $? KDBaseline

}

# KDBaseline
# train baseline


# LSTM_FC
# nohup python3 -u distill.py --use-mpo --mpo-type "lstm" "fc" --fc1-input-shape 10 2 3 10 --fc1-output-shape 6 2 2 8 --fc2-input-shape 6 2 2 8 --fc2-output-shape 5 1 1 1 --xh-input-shape 10 3 1 10 --xh-output-shape 10 3 4 10 --hh-input-shape 10 3 1 10 --hh-output-shape 10 3 4 10 >> $log_path 2>&1
# post $? LSTM_FC

# EMB_LSTM_FC
train EMB_LSTM_FC --use-mpo --mpo-type "lstm" "fc" "embedding" --fc1-input-shape 10 2 3 10 --fc1-output-shape 6 2 2 8 --fc2-input-shape 6 2 2 8 --fc2-output-shape 5 1 1 1 --xh-input-shape 10 3 1 10 --xh-output-shape 10 3 4 10 --hh-input-shape 10 3 1 10 --hh-output-shape 10 3 4 10 --embedding-input-shape 19 8 7 20 --embedding-output-shape 10 3 1 10

