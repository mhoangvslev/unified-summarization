if [ ! $# -eq 3 ]; then
    echo "Mismatch number of arguments. Given $#, required 3"
    echo "sh script/rewriter.sh <MODE> <path/to/data> <EXP_NAME>"
    exit 1
fi

MODE=$1; shift;
DATA_PATH=$(realpath $1/finished_files); shift;

TRAIN_PATH="$DATA_PATH/chunked/train_*"
VAL_PATH="$DATA_PATH/chunked/val_*"
TEST_PATH="$DATA_PATH/chunked/test_*"
VOCAB_PATH="$DATA_PATH/vocab"
EXP_NAME="$1"; shift;

MAX_ITER=10000
SAVE_MODEL_EVERY=1000
MAX_TO_KEEP=10

# Adjust to your hardware. Default is 256
HIDDEN_DIM=256

# for eval mode

DECODE_METHOD="beam"
START_EVAL=10000
INCONSISTENT_LOSS="True"
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

# for evalall mode
# for evalall mode
SAVE_PKL=False   # save the results in pickle files
SAVE_VIS=True  # save for visualization (this data is big)
LOAD_BEST_EVAL_MODEL=False
if [ $LOAD_BEST_EVAL_MODEL = "True" ]; then
  BEST_REWRITER=$(head -n 1 log/rewriter/$EXP_NAME/eval_$EVAL_METHOD/checkpoint_best | sed 's/[^0-9]*//g')
  CKPT_PATH="log/rewriter/$EXP_NAME/eval_$EVAL_METHOD/bestmodel-$BEST_REWRITER"
else
  CKPT_PATH="log/rewriter/$EXP_NAME/train/model.ckpt_cov-81000"
fi

if [ "$MODE" = "train" ]; then
  LAST_CKPT=$(head -n 1 log/rewriter/$EXP_NAME/train/checkpoint | sed 's/[^0-9]*//g');
  if [ -z $LAST_CKPT ]; then 
    LAST_CKPT=0 
  elif [ $LAST_CKPT -ge 81000 ]; then
    echo "Stop training at step #81000, exiting"
    exit 0
  fi

  # Resume training
  echo "Begin/Resume training..."
  while [ $LAST_CKPT -lt 81000 ]; do
    echo $LAST_CKPT    
    # 1-10000
    if [ $LAST_CKPT -ge 0 -a $LAST_CKPT -lt 10000 ]; then
      python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM --max_enc_steps=50 --max_dec_steps=15 --max_train_iter=$((10000-$LAST_CKPT)) --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
    # 10001-20000
    elif [ $LAST_CKPT -ge 10000 -a $LAST_CKPT -lt 20000 ]; then
      python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM --max_enc_steps=100 --max_dec_steps=25 --max_train_iter=$((20000-$LAST_CKPT)) --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
    # 20001-30000
    elif [ $LAST_CKPT -ge 20000 -a $LAST_CKPT -lt 30000 ]; then
      python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM --max_enc_steps=150 --max_dec_steps=40 --max_train_iter=$((30000-$LAST_CKPT)) --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
    # 30001-40000
    elif [ $LAST_CKPT -ge 30000 -a $LAST_CKPT -lt 40000 ]; then
      python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM --max_enc_steps=200 --max_dec_steps=50 --max_train_iter=$((40000-$LAST_CKPT)) --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
    # 40001-50000
    elif [ $LAST_CKPT -ge 40000 -a $LAST_CKPT -lt 50000 ]; then
      python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM --max_enc_steps=250 --max_dec_steps=60 --max_train_iter=$((50000-$LAST_CKPT)) --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
    # 50001-60000
    elif [ $LAST_CKPT -ge 50000 -a $LAST_CKPT -lt 60000 ]; then
      python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM --max_enc_steps=300 --max_dec_steps=80 --max_train_iter=$((60000-$LAST_CKPT)) --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
    # 60001-70000
    elif [ $LAST_CKPT -ge 60000 -a $LAST_CKPT -lt 70000 ]; then
      python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM --max_enc_steps=350 --max_dec_steps=100 --max_train_iter=$((70000-$LAST_CKPT)) --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
    # 70001-80000
    elif [ $LAST_CKPT -ge 70000 -a $LAST_CKPT -lt 80000 ]; then
      python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM --max_enc_steps=400 --max_dec_steps=100 --max_train_iter=$((80000-$LAST_CKPT)) --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
    # add coverage mechanism for 1000 iter
    elif [ $LAST_CKPT -ge 80000 -a $LAST_CKPT -lt 81000 ]; then
      python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM --max_enc_steps=400 --max_dec_steps=100 --coverage=True --convert_to_coverage_model=True
      python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM --max_enc_steps=400 --max_dec_steps=100 --max_train_iter=1000 --save_model_every=200 --coverage=True --model_max_to_keep=$MAX_TO_KEEP
    fi

    LAST_CKPT=$(head -n 1 log/rewriter/$EXP_NAME/train/checkpoint | sed 's/[^0-9]*//g');
    if [ -z $LAST_CKPT ]; then 
      echo "Fatal error, exiting..."
      exit 1
    fi
    
  done
elif [ "$MODE" = "eval" ]; then
  python main.py --model=rewriter --mode=eval --data_path=$VAL_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM --max_enc_steps=400 --max_dec_steps=120 --coverage=False --batch_size=64 --eval_method=$EVAL_METHOD --decode_method=$DECODE_METHOD --start_eval_rouge=$START_EVAL --save_model_every=$SAVE_MODEL_EVERY --single_pass=$SINGLE_PASS
elif [ "$MODE" = "decode" ]; then
  # decode
  rm -rf log/rewriter/$EXP_NAME/decode_test_* &&
  python main.py --model=rewriter --mode=evalall --data_path=$TEST_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --hidden_dim_rewriter=$HIDDEN_DIM --max_enc_steps=400 --max_dec_steps=120 --coverage=True --decode_method=beam --single_pass=1 --eval_method=$EVAL_METHOD --load_best_eval_model=$LOAD_BEST_EVAL_MODEL --eval_ckpt_path=$CKPT_PATH --save_vis=$SAVE_VIS --save_pkl=$SAVE_PKL
fi
