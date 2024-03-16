import torch 
from torch import distributed as dist
import math

# This implementation is courtesy of Xin Li, University of Toronto
# and Vector Institute for AI research

class StatefulSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data = data_source
        self.shuffle = shuffle

        # initial dataloader index
        self.init_index()

    def init_index(self):

        if self.shuffle:
            self.indices = torch.randperm(len(self.data))
        else:
            self.indices = torch.arange(len(self.data))

        self.data_counter = 0

    def __iter__(self):
        self.init_index()
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.data_counter == len(self.data):
            #self.init_index()
            raise StopIteration()
        else:
            ele = self.indices[self.data_counter]
            self.data_counter += 1
            return int(ele)

    def state_dict(self, dataloader_iter=None):
        prefetched_num = 0
        # in the case of multiworker dataloader, the helper worker could be
        # pre-fetching the data that is not consumed by the main dataloader.
        # we need to subtract the unconsumed part .
        if dataloader_iter is not None:
            if dataloader_iter._num_workers > 0:
                batch_size = dataloader_iter._index_sampler.batch_size
                prefetched_num = \
                    (dataloader_iter._send_idx - dataloader_iter._rcvd_idx) * batch_size
        return {
                'indices': self.indices,
                'data_counter': self.data_counter - prefetched_num,
            }

    def load_state_dict(self, state_dict):
        self.indices = state_dict['indices']
        self.data_counter = state_dict['data_counter']



class StatefulDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, num_replicas,
                 rank, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        self.init_index()

    def init_index(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        self.indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(self.indices) == self.num_samples
        self.data_counter = 0

    def __iter__(self):
        self.init_index()
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        if self.data_counter == len(self):
            raise StopIteration()
        else:
            ele = self.indices[self.data_counter]
            self.data_counter += 1
            return int(ele)

    def state_dict(self, dataloader_iter=None):
        prefetched_num = 0
        # in the case of multiworker dataloader, the helper worker could be
        # pre-fetching the data that is not consumed by the main dataloader.
        # we need to subtract the unconsumed part .
        if dataloader_iter is not None:
            if dataloader_iter._num_workers > 0:
                batch_size = dataloader_iter._index_sampler.batch_size
                prefetched_num = \
                    (dataloader_iter._send_idx - dataloader_iter._rcvd_idx) * batch_size
        return {
                'indices': self.indices,
                'data_counter': self.data_counter - prefetched_num,
            }

    def load_state_dict(self, state_dict):
        self.indices = state_dict['indices']
        self.data_counter = state_dict['data_counter']