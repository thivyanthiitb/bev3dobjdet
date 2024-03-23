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


def main():
    dist.init()

    parser = argparse.ArgumentParser()
    
    # Add a new argument for specifying weights
    parser.add_argument("--weights", default="none", choices=["none", "unibev", "metabev", "bevfusion"],
                    help="Specify the type of weights to load (none, unibev, metabev)")
    
    
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    
    

    
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    # logger.info(f"Config:\n{cfg.pretty_text}")
    cfg_dict = cfg.to_dict()
    # logger.info(f"Config:\n{pformat(cfg_dict, indent=4)}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])
            
    def load_weights(model, weights_type):
        if weights_type == 'unibev':
            # Insert logic to load UNIBEV weights
            # unibev_state_dict = model.state_dict()
            # bevfusion_state_dict = torch.load("pretrained/bevfusion-det.pth")["state_dict"]

            # model.load_state_dict(bevfusion_state_dict, strict=False)

            # unibev_state_dict["fuser.conv3x3.weight"] = bevfusion_state_dict["fuser.0.weight"]
            # unibev_state_dict["fuser.bnorm.weight"] = bevfusion_state_dict["fuser.1.weight"]
            # unibev_state_dict["fuser.bnorm.bias"] = bevfusion_state_dict["fuser.1.bias"]
            
            # model.load_state_dict(unibev_state_dict)
            
            pretrained_unibev_dict = torch.load("pretrained/unibev_epoch_1.pth")["state_dict"]
            model.load_state_dict(pretrained_unibev_dict, strict=False)
            
            # for param in model.parameters():
            #     param.requires_grad = False

            # for param in model.fuser.parameters():
            #     param.requires_grad = True
            print("unibev weights loaded")
        
        elif weights_type == 'bevfusion':
            # Insert logic to load 
            
            pretrained_dict = torch.load("pretrained/bevfusion-det.pth")["state_dict"]
            model.load_state_dict(pretrained_dict, strict=False)
            
            # for param in model.parameters():
            #     param.requires_grad = True

            # for param in model.fuser.parameters():
            #     param.requires_grad = True
            print("bevfusion weights loaded")
            
        elif weights_type == 'bevfusion_aug':
            # Insert logic to load 
            
            pretrained_dict = torch.load("test/convfuser/epoch_2.pth")["state_dict"]
            model.load_state_dict(pretrained_dict, strict=False)
            
            # for param in model.parameters():
            #     param.requires_grad = True

            # for param in model.fuser.parameters():
            #     param.requires_grad = True
            print("bevfusion weights loaded")
            
        elif weights_type == 'metabev':
            # Insert logic to load METABEV weights
            metabev_state_dict = model.state_dict()
            bevfusion_state_dict = torch.load("pretrained/bevfusion-det.pth")["state_dict"]

            model.load_state_dict(bevfusion_state_dict, strict=False)
            
            metabev_state_dict["fuser.fuser.0.weight"] = bevfusion_state_dict["fuser.0.weight"]
            metabev_state_dict["fuser.fuser.1.weight"] = bevfusion_state_dict["fuser.1.weight"]
            metabev_state_dict["fuser.fuser.1.bias"]   = bevfusion_state_dict["fuser.1.bias"]
            
            model.load_state_dict(metabev_state_dict, strict=False)

            for param in model.parameters():
                param.requires_grad = False

            for param in model.fuser.parameters():
                param.requires_grad = True

            for param in model.fuser.fuser.parameters():
                param.requires_grad = False
            print("metabev weights loaded")
        # You can add more conditions here for other types of weights

    # Use the new function to load weights based on the command-line argument
    load_weights(model, args.weights)

    # logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
