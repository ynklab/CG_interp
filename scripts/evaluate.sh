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
        mkdir -p results/base/${TGT}_base_${PATTERN}$HINT_$EPOCH/$SEED
        python3 src/main.py \
            --inference\
            --beam\
            --pos_enc default\
            --model_path models/base/${TGT}_${PATTERN}_models${HINT}/$SEED/epoch_${EPOCH}.pt\
            --src en\
            --tgt $TGT\
            --hint $HINT_\
            --pos_enc default\
            --enc_layers 3\
            --dec_layers 3\
            --attn_heads 4\
            --data_train_path data/base/$PATTERN\
            --data_path data/base/$PATTERN\
            --result_path results/base/${TGT}_base_${PATTERN}${HINT}_$EPOCH/$SEED\
            --epochs 30\
            --lr 1e-4\
            --batch 512\
            --embed_init xavier\
            --seed $SEED

        mkdir -p results/masked/${TGT}_mask_${PATTERN}${HINT}_${EPOCH}/$SEED
        python3 src/main.py \
                --inference\
                --beam\
                --prune\
                --pos_enc default\
                --model_path models/masked/${TGT}_${PATTERN}_models_masked${HINT}_$EPOCH/$SEED/epoch_300.pt\
                --src en\
                --tgt $TGT\
                --hint $HINT_\
                --pos_enc default\
                --enc_layers 3\
                --dec_layers 3\
                --attn_heads 4\
                --data_path data/base/$PATTERN\
                --data_train_path data/base/$PATTERN\
                --result_path results/masked/${TGT}_mask_${PATTERN}${HINT}_${EPOCH}/$SEED\
                --epochs 30\
                --lr 1e-4\
                --batch 512\
                --embed_init xavier\
                --seed $SEED
    done