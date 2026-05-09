import os
import torch
import torch.distributed as dist

import datetime

def setup_distributed():
    """
    初始化分布式环境 (DDP)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    print(f'| distributed init (rank {rank}): {dist_backend}', flush=True)
    dist.init_process_group(
        backend=dist_backend, 
        init_method='env://',
        world_size=world_size, 
        rank=rank,
        timeout=datetime.timedelta(seconds=3600)
    )
    dist.barrier()
    return True, rank, world_size, gpu

def cleanup():
    """
    销毁进程组
    """
    dist.destroy_process_group()

def is_main_process():
    """
    判断是否为主进程
    """
    return dist.get_rank() == 0 if dist.is_initialized() else True

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
