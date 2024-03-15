rm -rf ./build
python3 setup.py develop
cd mmdet3d/models/ops
bash make.sh