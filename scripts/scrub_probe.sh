PATTERN=$1 # dobjpp2iobjpp or objpp2subjpp
TGT=$2 # ja or cogs
EPOCH=$3
SEED=$4

for MODE in "leace_dependency" "leace_dependency_dobjmod" "leace_dependency_iobjmod" "leace_dependency_mod" "leace_constituency" "leace_constituency_dobjmod" "leace_constituency_iobjmod" "leace_constituency_mod"
do
    if [ $PATTERN == "objpp2subjpp" ] && [ $MODE == "leace_dependency_iobjmod" ]
        MODE="leace_dependency_subjmod"
    elif [ $PATTERN == "objpp2subjpp" ] && [ $MODE == "leace_constituency_iobjmod" ]
        MODE="leace_constituency_subjmod"
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
        mkdir -p models/scrub_probe/${TGT}_probe_classifier_scrub_${MODE}${TYPE}_${EPOCH}

        python3 src/scrub_probe.py \
                --data_path data/base/$PATTERN\
                --data_train_path data/base/$PATTERN\
                --span_data_path data/scrub/$PATTERN\
                --model_path models/base/${TGT}_${PATTERN}_models${HINT}/$SEED/epoch_${EPOCH}.pt\
                --logging_dir models/scrub_probe/${TGT}_probe_classifier_scrub_${MODE}${TYPE}_$EPOCH/\
                --scrub\
                --counterfactual\
                --probe_hint\
                --src en\
                --tgt $TGT\
                --hint $HINT_\
                --probe_mode $TYPE\
                --probe_layer $LAYER\
                --epochs 20\
                --enc_layers 3\
                --dec_layers 3\
                --attn_heads 4\
                --lr 5e-4\
                --batch 256\
                --embed_init xavier\
                --weight_decay 0.1\

        mkdir -p models/scrub_probe/${TGT}_probe_mask_classifier_scrub_${MODE}${TYPE}_${EPOCH}

        python3 src/scrub_probe.py \
                --data_path data/base/$PATTERN\
                --data_train_path data/base/$PATTERN\
                --span_data_path data/scrub/$PATTERN\
                --model_path models/masked/${TGT}_${PATTERN}_models_masked${HINT}_${EPOCH}/$SEED/epoch_300.pt\
                --logging_dir models/scrub_probe/${TGT}_probe_mask_classifier_scrub_${MODE}${TYPE}_$EPOCH/\
                --scrub\
                --counterfactual\
                --probe_hint\
                --is_masked\
                --src en\
                --tgt $TGT\
                --hint $HINT_\
                --probe_mode $TYPE\
                --probe_layer $LAYER\
                --epochs 20\
                --enc_layers 3\
                --dec_layers 3\
                --attn_heads 4\
                --lr 5e-4\
                --batch 256\
                --embed_init xavier\
                --weight_decay 0.1\
        
        mkdir -p results/scrub_probe/${TGT}_probe_mask_scrub_${PATTERN}_${MODE}${TYPE}_$EPOCH/$SEED

        python3 src/scrub_probe.py \
                --inference\
                --data_path data/base/$PATTERN\
                --data_train_path data/base/$PATTERN\
                --span_data_path data/scrub/$PATTERN\
                --model_path models/masked/${TGT}_${PATTERN}_models_masked${HINT}_${EPOCH}/$SEED/epoch_300.pt\
                --classifier_path models/scrub_probe/${TGT}_probe_mask_classifier_scrub_${MODE}${TYPE}_$EPOCH/epoch_20.pt\
                --result_path results/scrub_probe/${TGT}_probe_mask_scrub_${PATTERN}_${MODE}${TYPE}_$EPOCH/$SEED\
                --logging_dir models/scrub_probe/${TGT}_probe_mask_classifier_scrub_${MODE}${TYPE}_$EPOCH/\
                --scrub\
                --counterfactual\
                --probe_hint\
                --is_masked\
                --src en\
                --tgt $TGT\
                --hint $HINT_\
                --probe_mode $TYPE\
                --probe_layer $LAYER\
                --epochs 300\
                --enc_layers 3\
                --dec_layers 3\
                --attn_heads 4\
                --lr 5e-4\
                --batch 256\
                --embed_init xavier\
                --weight_decay 0.1\

        mkdir -p results/scrub_probe/${TGT}_probe_scrub_${PATTERN}_${MODE}${TYPE}_$EPOCH/$SEED

        python3 src/scrub_probe.py \
                --inference\
                --data_path data/base/$PATTERN\
                --data_train_path data/base/$PATTERN\
                --span_data_path data/scrub/$PATTERN\
                --model_path models/base/${TGT}_${PATTERN}_models${HINT}/$SEED/epoch_$EPOCH.pt\
                --classifier_path models/scrub_probe/${TGT}_probe_classifier_scrub_${MODE}${TYPE}_$EPOCH/epoch_20.pt\
                --result_path results/scrub_probe/${TGT}_probe_scrub_${PATTERN}_${MODE}${TYPE}_$EPOCH/$SEED\
                --logging_dir models/scrub_probe/${TGT}_probe_classifier_scrub_${MODE}${TYPE}_$EPOCH/\
                --scrub\
                --counterfactual\
                --probe_hint\
                --src en\
                --tgt $TGT\
                --hint $HINT_\
                --probe_mode $TYPE\
                --probe_layer $LAYER\
                --epochs 300\
                --enc_layers 3\
                --dec_layers 3\
                --attn_heads 4\
                --lr 5e-4\
                --batch 256\
                --embed_init xavier\
                --weight_decay 0.1\

    done
done