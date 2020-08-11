if [ ! $# -eq 3 ]; then
    echo "Mismatch number of arguments. Given $#, required 3"
    echo "sh script/end2end.sh <MODE> <path/to/data> <EXP_NAME>"
    exit 1
fi

MODE=$1; shift;
DATA_PATH=$(realpath $1/finished_files); shift;

TRAIN_PATH="$DATA_PATH/chunked/train_*"
VAL_PATH="$DATA_PATH/chunked/val_*"
TEST_PATH="$DATA_PATH/chunked/test_*"
VOCAB_PATH="$DATA_PATH/vocab"
EXP_NAME="$1"; shift;

MAX_ITER=80000
SAVE_MODEL_EVERY=1000
MAX_TO_KEEP=5

BATCH_SIZE=4 # Adjustable to adapt to the hardware

# Adjustable to adapt to the hardware. Default is 200
HIDDEN_DIM_SELECTOR=$(grep -m 1 "HIDDEN_DIM=" scripts/selector.sh | sed 's/[^0-9]*//g') 

# Adjustable to adapt to the hardware. Default is 256
HIDDEN_DIM_REWRITER=$(grep -m 1 "HIDDEN_DIM=" scripts/rewriter.sh | sed 's/[^0-9]*//g') 

MAX_ART_LEN=50
MAX_ENC_STEPS=600
LR=0.01
SELECTOR_LOSS_WT=5.0
INCONSISTENT_LOSS="True"
INCONSISTENT_TOPK=3
UPDATE_ATTENTION="False"

# for eval mode
DECODE_METHOD="greedy"
START_EVAL=1000
case "$INCONSISTENT_LOSS" in
  "True"*)
    SINGLE_PASS="False"
    EVAL_METHOD="loss"
    ;;
  "False"*)
    SINGLE_PASS="True"
    EVAL_METHOD="rouge"
    ;;
esac

BEST_SELECTOR=$(head -n 1 log/selector/$EXP_NAME/eval/checkpoint_best | sed 's/[^0-9]*//g')
SELECTOR_PATH="log/selector/$EXP_NAME/eval/bestmodel-$BEST_SELECTOR"

# BEST_REWRITER=$(head -n 1 log/rewriter/$EXP_NAME/eval_$EVAL_METHOD/checkpoint_best | sed 's/[^0-9]*//g')
# REWRITER_PATH="log/rewriter/$EXP_NAME/eval_$EVAL_METHOD/bestmodel-$BEST_REWRITER"
REWRITER_PATH="log/rewriter/$EXP_NAME/train/model.ckpt_cov-81000"

# for evalall mode
SAVE_PKL=True   # save the results in pickle files
SAVE_VIS=False  # save for visualization (this data is big)
LOAD_BEST_EVAL_MODEL=True 

if [ $LOAD_BEST_EVAL_MODEL = "True" ]; then
  BEST_END2END=$(head -n 1 log/end2end/$EXP_NAME/eval_$EVAL_METHOD/checkpoint_best | sed 's/[^0-9]*//g')
  CKPT_PATH="log/end2end/$EXP_NAME/eval_$EVAL_METHOD/bestmodel-$BEST_END2END"
else
  BEST_END2END=$(head -n 1 log/end2end/$EXP_NAME/train/checkpoint | sed 's/[^0-9]*//g')
  CKPT_PATH="log/end2end/$EXP_NAME/train/model.ckpt_cov-$BEST_END2END"
fi

LAST_CKPT=$(head -n 1 log/end2end/$EXP_NAME/train/checkpoint | sed 's/[^0-9]*//g')
if [ -z $LAST_CKPT ]; then
  LAST_CKPT=0;
fi

if [ "$MODE" = "train" ]; then
  python main.py --model=end2end --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=$MAX_ENC_STEPS --max_dec_steps=100 --max_train_iter=$MAX_ITER --batch_size=$BATCH_SIZE --max_art_len=$MAX_ART_LEN --lr=$LR --selector_loss_wt=$SELECTOR_LOSS_WT --inconsistent_loss=$INCONSISTENT_LOSS --inconsistent_topk=$INCONSISTENT_TOPK --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP --coverage=True --update_attention=$UPDATE_ATTENTION --pretrained_selector_path=$SELECTOR_PATH --pretrained_rewriter_path=$REWRITER_PATH
elif [ "$MODE" = "eval" ]
then
  python main.py --model=end2end --mode=eval --data_path=$VAL_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM_REWRITER --hidden_dim_selector=$HIDDEN_DIM_SELECTOR --max_enc_steps=$MAX_ENC_STEPS --max_dec_steps=120 --max_art_len=$MAX_ART_LEN --batch_size=64 --selector_loss_wt=$SELECTOR_LOSS_WT --inconsistent_loss=$INCONSISTENT_LOSS --inconsistent_topk=$INCONSISTENT_TOPK --eval_method=$EVAL_METHOD --decode_method=$DECODE_METHOD --start_eval_rouge=$START_EVAL --save_model_every=$SAVE_MODEL_EVERY --single_pass=$SINGLE_PASS --coverage=True --update_attention=$UPDATE_ATTENTION
elif [ "$MODE" = "decode" ]
then
  python main.py --model=end2end --mode=evalall --data_path=$TEST_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM_REWRITER --hidden_dim_selector=$HIDDEN_DIM_SELECTOR --max_enc_steps=$MAX_ENC_STEPS --max_dec_steps=120 --max_art_len=$MAX_ART_LEN --decode_method=beam --coverage=True --single_pass=1 --save_pkl=$SAVE_PKL --save_vis=$SAVE_VIS --inconsistent_loss=$INCONSISTENT_LOSS --inconsistent_topk=$INCONSISTENT_TOPK --eval_ckpt_path=$CKPT_PATH --eval_method=$EVAL_METHOD --load_best_eval_model=$LOAD_BEST_EVAL_MODEL --update_attention=$UPDATE_ATTENTION
fi
