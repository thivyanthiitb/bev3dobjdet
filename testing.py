import argparse
import copy
import os
import random
import time
#extra
from pprint import pformat

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

bevfusion_state_dict = torch.load("pretrained/bevfusion-det.pth")["state_dict"]
print(bevfusion_state_dict)