python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/metafuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --data.samples_per_gpu 2 &>> all_logs/txts/240319_1000.txt