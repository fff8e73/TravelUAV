import os

import torch
import torch.distributed as dist

# 尝试导入完整的参数配置，如果失败则使用None
# 这样可以让utils在没有transformers的情况下也能工作
try:
    from src.common.param import args
except (ImportError, ModuleNotFoundError):
    # 如果导入失败，args将在后续被设置
    args = None


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        if args is not None:
            args.DistributedDataParallel = False
        return

    if args is not None:
        args.DistributedDataParallel = True

    torch.cuda.set_device(gpu)
    print('distributed init (rank {}, word {})'.format(rank, world_size), flush=True)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    torch.distributed.barrier()


def manual_init_distributed_mode(rank, world_size, local_rank):
    if args is not None:
        args.DistributedDataParallel = True
        args.batchSize = 1

    gpu = local_rank
    torch.cuda.set_device(gpu)

    os.environ['MASTER_ADDR'] = 'localhost'
    if args is not None:
        os.environ['MASTER_PORT'] = str(args.DDP_MASTER_PORT)
    else:
        os.environ['MASTER_PORT'] = '20001'  # 默认端口
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)

    print('distributed init (rank {}, word {})'.format(rank, world_size), flush=True)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    torch.distributed.barrier()


def FromPortGetPid(port: int):
    import subprocess
    import time
    import signal

    subprocess_execute = "netstat -nlp | grep {}".format(
        port,
    )

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print(
            "{}\t{}\t{}".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                'FromPortGetPid',
                e,
            )
        )
        return None
    except:
        return None

    pid = None
    for line in iter(p.stdout.readline, b''):
        line = str(line, encoding="utf-8")
        if 'tcp' in line:
            pid = line.strip().split()[-1].split('/')[0]
            try:
                pid = int(pid)
            except:
                pid = None
            break

    try:
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    return pid

