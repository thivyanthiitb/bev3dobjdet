#torchpack dist-run -np 2 python tools/test.py configs/robodrive/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox
# python tools/test.py configs/robodrive/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml runs/run-71c2c712/epoch_1.pth --eval bbox
# python tools/test.py configs/robodrive/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml runs/run-71c2c712/latest.pth --eval bbox
#python tools/test.py configs/nuscenes_eval/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml runs/run-71c2c712/epoch_1.pth --eval bbox

python tools/test.py configs/robodrive/det/transfusion/secfpn/camera+lidar/swint_v0p075/unifuser.yaml test/unifuser/epoch_1.pth --eval bbox
