import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Dataset:
    def __getitem__(self, i):
        return i 
    def __len__(self): 
        return 40


from sampler import StatefulDistributedSampler, StatefulSampler
dataset = Dataset()
sampler = StatefulSampler(dataset)
# sampler = StatefulDistributedSampler(dataset, 2, 0, shuffle=False, seed=0, drop_last=False)


from torch.utils.data import DataLoader



loader = DataLoader(dataset, batch_size=10, num_workers=2, sampler=sampler)
loader_iter = iter(loader)
batch = next(loader_iter)
batch = next(loader_iter)

sd = sampler.state_dict(loader_iter)
print(sd)

