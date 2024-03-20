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
    torchpack dist-run -np 2 python tools/train.py \
        --weights unibev \
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/unifuser.yaml \
        --dataset_type NuScenesDataset \
        --dataset_root data/nuscenes/ \
        --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
        --data.samples_per_gpu 12 \
        --max_epochs 6 \
        2>&1 | tee all_logs/txts/240320_1100_forcefuse_unibev.txt
else
    echo "Invalid argument. Please specify '1' for the first command or '2' for the second."
fi
