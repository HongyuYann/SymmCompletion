import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.config import *
from utils import parser, dist_utils, misc
from thop import profile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def Complexity(net, config):
    B = 1
    N = 2048
    C = 3
    model = config.model.NAME

    # test FLOPs and #Params
    points = torch.rand(B, N, C).float().cuda()
    depth = torch.rand(B*3, 1, 224, 224).float().cuda()

    # depth[:, :1, :, :] = depth * 2. - 1
    total_time = 0
    for _ in range(20):
        inputs = torch.rand(B, N, C).float().cuda()  # range: [0, 1]
        inputs[:, :, :3] = inputs * 2. - 1.  # xyz range: [-1, 1]
        for _ in range(20):
            start = time.perf_counter()
            if model == "SVDFormer":
                ret = net(inputs, depth)
            else:
                ret = net(inputs)
            total_time += (time.perf_counter() - start)
    avg_speed = total_time * 1000 / (B * 20 * 20)
    if model == "SVDFormer":
        macs, params = profile(net, inputs=(points,depth))
    else:
        macs, params = profile(net, inputs=(points,))
    flops = macs * 2 / B
    print(f'{model}: FLOPs-{flops} or {flops/1e9}G, #Params-{params} or {params/1e6}M, Inference speed-{avg_speed} (ms/sample)')
    

if __name__ == '__main__':

    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size

    log_file = os.path.join(args.experiment_path, f'complexity.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    # config
    config = get_config(args, logger = logger)
    print_log('Parameters start ... ', logger = logger)
 
    base_model = builder.model_builder(config.model)
    base_model.cuda()
    base_model.eval()
    Complexity(base_model, config)
