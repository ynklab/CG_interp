PATTERN=$1 # dobjpp2iobjpp or objpp2subjpp
TGT=$2 # ja or cogs
EPOCH=$3
SEED=$4

for MODE in "leace_dependency" "leace_dependency_dobjmod" "leace_dependency_iobjmod" "leace_dependency_mod" "leace_constituency" "leace_constituency_dobjmod" "leace_constituency_iobjmod" "leace_constituency_mod"
do
    if [ $PATTERN == "objpp2subjpp" ] && [ $MODE == "leace_dependency_iobjmod" ]
    then
        MODE="leace_dependency_subjmod"
    elif [ $PATTERN == "objpp2subjpp" ] && [ $MODE == "leace_constituency_iobjmod" ]
    then
        MODE="leace_constituency_subjmod"
    else
    then
        MODE=$MODE
    fi
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
        MODE_=$MODE
        mkdir -p models/scrub/${TGT}_${PATTERN}_scrub_sub_${MODE_}${HINT}/$SEED
        mkdir -p results/scrub/${TGT}_scrub_${PATTERN}_mask_${MODE_}${HINT}_${EPOCH}/$SEED
        python3 src/scrub.py \
                --data_path data/base/$PATTERN\
                --data_train_path data/base/$PATTERN\
                --span_data_path data/scrub/$PATTERN\
                --model_path models/masked/${TGT}_${PATTERN}_models_masked${HINT}_${EPOCH}/$SEED/epoch_300.pt\
                --logging_dir models/scrub/${TGT}_${PATTERN}_scrub_sub_${MODE_}${HINT}/$SEED/\
                --result_path results/scrub/${TGT}_scrub_${PATTERN}_mask_${MODE_}${HINT}_${EPOCH}/$SEED\
                --is_masked\
                --counterfactual\
                --scrub\
                --src en\
                --tgt $TGT\
                --hint $HINT_\
                --probe_mode $MODE\
                --probe_hint\
                --enc_layers 3\
                --dec_layers 3\
                --attn_heads 4\
                --batch 256\
                --embed_init xavier\
                --seed $SEED

        mkdir -p models/scrub/${TGT}_${PATTERN}_scrub_${MODE_}${HINT}/$SEED
        mkdir -p results/scrub/${TGT}_scrub_${PATTERN}_${MODE_}${HINT}_${EPOCH}/$SEED
        python3 src/scrub.py \
                --data_path data/base/$PATTERN\
                --data_train_path data/base/$PATTERN\
                --span_data_path data/scrub/$PATTERN\
                --model_path models/base/${TGT}_${PATTERN}_models${HINT}/$SEED/epoch_${EPOCH}.pt\
                --logging_dir models/scrub/${TGT}_${PATTERN}_scrub_${MODE_}${HINT}/$SEED/\
                --result_path results/scrub/${TGT}_scrub_${PATTERN}_${MODE_}${HINT}_${EPOCH}/$SEED\
                --counterfactual\
                --scrub\
                --src en\
                --tgt $TGT\
                --hint $HINT_\
                --probe_mode $MODE\
                --probe_hint\
                --enc_layers 3\
                --dec_layers 3\
                --attn_heads 4\
                --batch 256\
                --embed_init xavier\
                --seed $SEED
    done
done