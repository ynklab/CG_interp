PATTERN=$1 # dobjpp2iobjpp or objpp2subjpp
TGT=$2 # ja or cogs
SEED=$3

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
    mkdir -p models/base/${TGT}_${PATTERN}_models$HINT/$SEED
    python src/main.py \
        --src en\
        --tgt $TGT\
        --hint $HINT_\
        --data_path data/base/$PATTERN\
        --data_train_path data/base/$PATTERN\
        --logging_dir models/base/${TGT}_${PATTERN}_models$HINT/$SEED/\
        --epochs 500\
        --enc_layers 3\
        --dec_layers 3\
        --attn_heads 4\
        --lr 1e-4\
        --batch 256\
        --seed $SEED
done