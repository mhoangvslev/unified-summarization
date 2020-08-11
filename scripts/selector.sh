if [ ! $# -eq 3 ]; then
    echo "Mismatch number of arguments. Given $#, required 3"
    echo "sh script/selector.sh <MODE> <path/to/data> <EXP_NAME>"
    exit 1
fi

MODE=$1; shift;
DATA_PATH=$(realpath $1/finished_files); shift;

TRAIN_PATH="$DATA_PATH/chunked/train_*"
VAL_PATH="$DATA_PATH/chunked/val_*"
TEST_PATH="$DATA_PATH/chunked/test_*"
VOCAB_PATH="$DATA_PATH/vocab"

EXP_NAME="$1"; shift;

# for train mode

# Adapt to your hardware. Default is 200
HIDDEN_DIM=200 
MAX_ITER=50000
BATCH_SIZE=32
SAVE_MODEL_EVERY=1000
MAX_TO_KEEP=3
#PRETRAINED=""  # uncomment this if you have pretrained selector model

# for evalall mode
SELECT_METHOD="prob"
MAX_SELECT=30
THRES=0.5
SAVE_PKL=True

LOAD_BEST_EVAL_MODEL=False
if [ $LOAD_BEST_EVAL_MODEL = "True" ]; then
  BEST_SELECTOR=$(head -n 1 log/selector/$EXP_NAME/eval/checkpoint_best | sed 's/[^0-9]*//g')
  CKPT_PATH="log/selector/$EXP_NAME/eval/bestmodel-$BEST_SELECTOR"
else
  BEST_SELECTOR=$(head -n 1 log/selector/$EXP_NAME/train/checkpoint | sed 's/[^0-9]*//g')
  CKPT_PATH="log/selector/$EXP_NAME/train/model.ckpt-$BEST_SELECTOR"
fi

LAST_CKPT=$(head -n 1 log/selector/$EXP_NAME/train/checkpoint | sed 's/[^0-9]*//g')
if [ -z $LAST_CKPT ]; then
  LAST_CKPT=0;
fi

if [ "$MODE" = "train" ]; then
  python main.py --model=selector --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_art_len=50 --max_sent_len=50 --max_train_iter=$MAX_ITER --batch_size=$BATCH_SIZE --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP #--pretrained_selector_path=$PRETRAINED
elif [ "$MODE" = "eval" ]
then
  python main.py --model=selector --mode=eval --data_path=$VAL_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_selector=$HIDDEN_DIM --max_art_len=50 --max_sent_len=50 --batch_size=$BATCH_SIZE
elif [ "$MODE" = "decode" ]
then
  rm -rf log/selector/$EXP_NAME/select_test_* &&
  python main.py --model=selector --mode=evalall --data_path=$TEST_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_selector=$HIDDEN_DIM --max_art_len=50 --max_sent_len=50 --max_select_sent=$MAX_SELECT --single_pass=True --select_method=$SELECT_METHOD --thres=$THRES --save_pkl=$SAVE_PKL --eval_ckpt_path=$CKPT_PATH --load_best_eval_model=$LOAD_BEST_EVAL_MODEL
fi
