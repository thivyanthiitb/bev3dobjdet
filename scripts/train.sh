#!/bin/bash

# Check for the first argument
if [ "$1" == "1" ]; then
    echo "Running first training command..."
    python tools/train.py \
        --weights unibev \
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/unifuser.yaml \
        --dataset_type NuScenesDataset \
        --dataset_root data/nuscenes/ \
        --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
        --data.samples_per_gpu 2 \
        --max_epochs 1 \
        2>&1 | tee all_logs/txts/240320_1100_forcefuse_unibev.txt
elif [ "$1" == "2" ]; then
    echo "Running second training command..."
    CUDA_VISIBLE_DEVICES=1
    torchpack dist-run -np 1 python tools/train.py \
        --weights bevfusion \
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser_aug.yaml \
        --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
        --dataset_type NuScenesDataset \
        --dataset_root data/nuscenes/ \
        --data.samples_per_gpu 9 \
        --max_epochs 1 \
        --reduce_beams 0 \
        2>&1 | tee all_logs/240323_0700_convfuser_aug_from_epoch2.txt \
        --load_augmented mvp \
else
    echo "Invalid argument. Please specify '1' for the first command or '2' for the second."
fi
