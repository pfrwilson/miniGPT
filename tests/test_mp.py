import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print('process group initialized')
    print(f'{dist.is_initialized()=}')
    
    dist.barrier()
    print('barrier passed')
    with torch.cuda.device(rank):
        A = torch.tensor([rank]).cuda(non_blocking=True)
        target = [torch.zeros_like(A) for _ in range(world_size)]
        dist.all_gather(target, A)
        print(target)
        
    dist.destroy_process_group()
    print('done!')


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    rank = int(os.getenv('SLURM_PROCID')) 
    world_size=int(os.getenv('SLURM_NTASKS'))
    print(f'{rank=}, {world_size=}')
    worker(rank, world_size)
