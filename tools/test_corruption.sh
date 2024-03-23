#torchpack dist-run -np 2 python tools/test.py configs/robodrive/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox
# python tools/test.py configs/robodrive/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml runs/run-71c2c712/epoch_1.pth --eval bbox
# python tools/test.py configs/robodrive/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml runs/run-71c2c712/latest.pth --eval bbox
#python tools/test.py configs/nuscenes_eval/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml runs/run-71c2c712/epoch_1.pth --eval bbox

torchpack dist-run -np 1 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml runs/run-90d1ae7d-51da0336/epoch_2.pth --eval bbox
torchpack dist-run -np 1 python tools/test.py \
    configs/robodrive/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    runs/run-90d1ae7d-51da0336/epoch_2.pth \
    --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
    --eval bbox
