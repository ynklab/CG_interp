PATTERN=$1 # dobjpp2iobjpp or objpp2subjpp
TGT=$2 # ja or cogs
EPOCH=$3
SEED=$4
for i in 0 1
do
    if [ $i -eq 0 ]
    then
        HINT=""
        HINT_="_"
    else
        HINT="_hint"
        HINT_="_hint"
    fi
    mkdir -p models/masked/${TGT}_${PATTERN}_models_masked${HINT}_${EPOCH}/$SEED/

    python src/main.py \
        --prune\
        --model_path models/base/${TGT}_${PATTERN}_models$HINT/$SEED/epoch_${EPOCH}.pt\
        --src en\
        --tgt $TGT\
        --hint $HINT_\
        --data_path data/base/${PATTERN}\
        --data_train_path data/base/${PATTERN}\
        --logging_dir models/masked/${TGT}_${PATTERN}_models_masked${HINT}_${EPOCH}/$SEED/\
        --epochs 300\
        --enc_layers 3\
        --dec_layers 3\
        --attn_heads 4\
        --embed_init xavier\
        --lr 5e-4\
        --batch 256\
        --lambda_init 1\
        --seed $SEED
done