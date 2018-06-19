CUDA_VISIBLE_DEVICES=0 python3 main_avito.py \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --net_type vggnet \
    --depth 16 \
    --resetClassifier \
    --finetune
