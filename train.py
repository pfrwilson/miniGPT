"""
Train a model to generate text.

Author: Paul Wilson 2023
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import RNN
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb
from dataclasses import dataclass, field
import coolname
from os.path import join
import os
import typing as tp
from datetime import datetime
import sys
import torch
import rich 
import json
import logging
import simple_parsing
from dataset import dataset_registry
from training_logic import registry as model_registry, TrainingLogic
import torch.distributed as dist
if not torch.cuda.is_available(): 
    raise RuntimeError('Could not find gpu.')


@dataclass
class Config:
    id: str # A unique id for the experiment
    group: str = 'default' # A group for the experiments
    checkpoint_dir: str | None = None # Directory to store checkpoints. If None, one will be created automatically. 
    distributed_training: bool = False # whether to use multiprocessing

    wandb: bool = True
    debug: bool = False

    lr: float = 0.001
    batch_size: int = 256
    n_epochs: int = 100
    model: model_registry.BaseConfig = simple_parsing.subgroups(
        {
            name: model_registry.get_config(name) for name in model_registry.list_constructibles()     
        }, default='premade'
    )
    model_path: str | None = None  # optional path to a model to load
    vocab_path: str = "data/vocab_1024.json"  # optional path to a vocab to load

    dataset: dataset_registry.BaseConfig = simple_parsing.subgroups(
        {name: dataset_registry.get_config(name) for name in dataset_registry.list_constructibles()}, 
        default=dataset_registry.list_constructibles()[0])

    exp_dir = None 


class Trainer: 
    def __init__(self, args: Config): 
        self.args = args 

    def setup(self): 
        # First thing we do is check for distributed training
        if args.distributed_training: 
            VALID_ENVIRONMENT = (
                'MASTER_ADDR' in os.environ, 
                'MASTER_PORT' in os.environ, 
                'RANK' in  os.environ, 
                'WORLD_SIZE' in os.environ
            )
            assert VALID_ENVIRONMENT, f'We need to have the necessary environment variables to run multiprocessing.'
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            dist.init_process_group('nccl', rank=self.rank, world_size=self.world_size)
        else:
            self.rank = 0 
            self.world_size = 1 

        # =========== BASIC LOGGING ================
        self.logger = logging.getLogger('Trainer')
        # format the logger
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if self.rank == 0: 
            self.logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
        else: 
            self.logger.setLevel(logging.CRITICAL)
        self.logger.info("Beginning setup.")

        # ============ EXPERIMENT DIRECTORY ============
        self.logger.info("Setting up experiment directory.")

        if self.rank == 0: 
            with open('exp_dir_lookup.json') as f: 
                exp_dir_lookup = json.load(f)
            if self.args.id in exp_dir_lookup: 
                self.logger.info(f"Found experiment directory for id {self.args.id}.")
                self.args.exp_dir = exp_dir_lookup[self.args.id]
            else: 
                self.logger.info(f"Could not find experiment directory for id {self.args.id}.")
                self.logger.info(f"Creating new experiment directory.")
                generated_exp_dir = join(
                    "logs", self.args.group, f"{datetime.now().strftime('%Y-%m-%d_%H:%M')}_{coolname.generate_slug(2)}"
                )
                self.args.exp_dir = generated_exp_dir
                self.logger.info(f"New experiment directory: {generated_exp_dir}")
                os.makedirs(generated_exp_dir)
                if self.args.checkpoint_dir is not None: 
                    os.symlink(
                        self.args.checkpoint_dir,
                        join(generated_exp_dir, 'checkpoints'),
                        target_is_directory=True,
                    )
                else: 
                    os.makedirs(join(generated_exp_dir, 'checkpoints'))
                exp_dir_lookup[self.args.id] = generated_exp_dir
                with open('exp_dir_lookup.json', 'w') as f: 
                    json.dump(exp_dir_lookup, f)
                    self.logger.info(f"Saved experiment directory lookup to exp_dir_lookup.json.")
                # TODO - make config serializable
                # with open(join(args.exp_dir, 'config.json'), 'w') as f: 
                #     self.logger.info(f"Saved experiment config to {join(args.exp_dir, 'config.json')}.")
                #     json.dump(vars(args), f)            

        if self.args.distributed_training: 
            dist.barrier() # we need to make sure the other processes wait for the above to be done

        # now everyone looks up the experiment directory. 
        with open('exp_dir_lookup.json') as f: 
            exp_dir_lookup = json.load(f)
            if args.id in exp_dir_lookup: 
                args.exp_dir = exp_dir_lookup[args.id]

        # =========== ADVANCED LOGGING ================
        # we also log to a file 
        self.logger.info("Setting up file logging.")
        if self.rank == 0: 
            file_handler = logging.FileHandler(join(args.exp_dir, 'out.log'))
            file_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(
                file_handler
            )

        # we also log to wandb
        
        if self.rank == 0 and args.wandb and not args.debug: 
            self.logger.info("Setting up wandb logging.")
            if args.wandb and not args.debug:
                wandb.init(
                    project="auto-preacher", config=vars(args), name=os.path.basename(args.exp_dir), group=args.group,
                )
        else: 
            self.logger.info("Not setting up wandb logging.")

        # =========== DATASET ==================
        self.logger.info("Setting up dataset.")
        self.train_loader, self.test_loader = self.create_dataloaders()
        self.logger.info(f"Created dataloaders with {len(self.train_loader)} train batches and {len(self.test_loader)} test batches.")

        # =========== MODEL ================== 
        self.logger.info("Setting up tokenizer.")
        from torchzero.utils.tokenizer import Tokenizer
        self.tokenizer = Tokenizer.from_json(args.vocab_path)

        self.logger.info("Setting up model.")
        self.model, self.training_logic = model_registry.build(args.model, tokenizer=self.tokenizer)
        self.training_logic: TrainingLogic
        self.model.to(self.rank)
        if self.args.distributed_training: 
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank])
        self.logger.info(f"Built model {self.model.__class__}.")

        # =========== OPTIMIZER, SCALER, SCHEDULER ==================
        self.logger.info("Setting up optimizer.")
        self.opt = optim.Adam(self.model.parameters(), lr=args.lr)
        self.logger.info(f"Using optimizer {self.opt.__class__}.")
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", patience=5, verbose=True
        )
        self.logger.info(f"Using scheduler {self.scheduler.__class__}.")
        self.best_val_loss = np.inf

        from torch.cuda.amp.grad_scaler import GradScaler
        self.scaler = GradScaler()
        self.logger.info(f"Using gradient scaler {self.scaler.__class__}.")
        self.logger.info("Setup complete.")

    def train(self):
        self.logger.info("Beginning training.")
        for epoch in range(1, args.n_epochs):
            t1 = datetime.now()
            train_loss = self.train_epoch(self.train_loader, self.model, self.opt, self.scaler)
            val_loss = self.val_epoch(self.test_loader, self.model)
            self.scheduler.step(val_loss)
            t2 = datetime.now()

            msg =f"""
=========== EPOCH {epoch} =========
TIME ELAPSED: {t2 - t1} 
TRAIN LOSS: {train_loss}
VAL LOSS: {val_loss}
GENERATED TEXT:


{self.training_logic.predict(self.model, self.tokenizer, self.rank, prompt=None, max_len=500)}


========================="""
            self.logger.info(msg)

            self.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "lr": self.opt.param_groups[0]["lr"],
                }
            )

            if val_loss < self.best_val_loss:
                self.logger.info(f"New best val loss: from {self.best_val_loss} to {val_loss}.")
                self.best_val_loss = val_loss
                if self.rank == 0:
                    torch.save(self.model.state_dict(), join(args.checkpoint_dir, "best_model.pt"))
                    if args.wandb:
                        wandb.run.summary["best_val_loss"] = self.best_val_loss
                    self.logger.info("Saved model!")

    def train_epoch(self, loader, model, optimizer, scaler):
        loss_epoch = 0
        total = 0
        model.train()

        for iteration, batch in enumerate(tqdm(loader, leave=False)):
            with torch.autocast('cuda'):
                loss = self.training_logic.step(
                    model, self.tokenizer, batch, self.rank
                )
            
            total += 1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_epoch += loss.item()

        return loss_epoch / total

    @torch.no_grad()
    def val_epoch(self, loader, model):
        loss_epoch = 0
        total = 0
        model.eval()

        for iteration, batch in enumerate(tqdm(loader, leave=False)):
            loss = self.training_logic.step(model, self.tokenizer, batch, self.rank)
            total += 1

            loss_epoch += loss.item()

        return loss_epoch / total

    def create_dataloaders(self):
        args = self.args
        # from dataset import TextFileDataset
        import torch

        #dataset = TextFileDataset("data/bible.txt", chunk_length=args.chunk_length)
        dataset = dataset_registry.build(args.dataset)
        from torch.utils.data import Subset

        train_dataset = Subset(dataset, range(int(len(dataset) * 0.8)))
        test_dataset = Subset(dataset, range(int(len(dataset) * 0.8), len(dataset)))

        from torch.utils.data.sampler import RandomSampler
        from torch.utils.data.distributed import DistributedSampler
        if args.distributed_training:
            train_sampler = DistributedSampler(train_dataset)
            test_sampler = DistributedSampler(test_dataset)
        else:
            train_sampler = RandomSampler(train_dataset)
            test_sampler = RandomSampler(test_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, sampler=test_sampler
        )
        return train_loader, test_loader

    def log(self, d): 
        if not self.rank == 0: 
            return 
        if self.args.wandb: 
            wandb.log(d)
        else: 
            logging.info(d)



""" def train(args: Config):
    # if args.world_size is not None: 
    #     using_multiprocessing = True
    #     assert 'MASTER_PORT' in os.environ and 'MASTER_ADDR' in os.environ, \
    #         "You need to pass MASTER_PORT and MASTER_ADDR as environment variables to use multi-gpu training."
    #     dist.init_process_group('nccl', rank=args.device, world_size=args.world_size)
    # else:
    #     using_multiprocessing = False 
    # master_process = using_multiprocessing and args.rank == 0 or not using_multiprocessing
    
    # if master_process: 
    #     logging.basicConfig(level=logging.INFO)
    #     os.makedirs(args.exp_dir, exist_ok=True)
    #     if args.wandb and not args.debug:
    #         wandb.init(
    #             project="auto-preacher", config=vars(args), name=os.path.basename(args.exp_dir), group=args.group,
    #         )
    #     else: 
    #         wandb.log = lambda x: rich.print(x)

    # if args.debug: 
    #     logging.basicConfig(
    #         level=logging.DEBUG
    #     )

    # Dataset
    train_loader, test_loader = create_dataloaders(args)

    # Tokenizer
    from torchzero.utils.tokenizer import Tokenizer
    tokenizer = Tokenizer.from_json(args.vocab_path)

    # Model
    model: TrainableModel = model_registry.build(args.model, tokenizer=tokenizer).to(args.device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # Optimizer
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=5, verbose=True
    )
    best_val_loss = np.inf

    # gradient scaler 
    from torch.cuda.amp.grad_scaler import GradScaler
    scaler = GradScaler()

    for epoch in range(1, args.n_epochs):
        train_loss = train_epoch(train_loader, model, tokenizer, opt, scaler)
        val_loss = val_epoch(test_loader, model, tokenizer)

        scheduler.step(val_loss)

        print(f"===== EPOCH {epoch} ======")
        print(f"LOSSES: ")
        print(f"TRAIN: {train_loss}")
        print(f"VAL: {val_loss}")
        print(f"GENERATED TEXT: ")
        print(model.predict(prompt=None, max_len=500))
        print()
        print("=========================")

        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch,
                "lr": opt.param_groups[0]["lr"],
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), join(args.exp_dir, "best_model.pt"))

            if args.wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss

            print("Saved model!")


def create_dataloaders(args: Config):
    # from dataset import TextFileDataset
    import torch

    #dataset = TextFileDataset("data/bible.txt", chunk_length=args.chunk_length)
    dataset = dataset_registry.build(args.dataset)
    from torch.utils.data import Subset

    train_dataset = Subset(dataset, range(int(len(dataset) * 0.8)))
    test_dataset = Subset(dataset, range(int(len(dataset) * 0.8), len(dataset)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    return train_loader, test_loader


def train_epoch(loader, model, tokenizer, optimizer, scaler):
    loss_epoch = 0
    total = 0
    model.train()

    for iteration, batch in enumerate(tqdm(loader, leave=False)):
        with torch.autocast('cuda'):
            loss = model.step(batch)
        
        total += 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss_epoch += loss.item()

    return loss_epoch / total


@torch.no_grad()
def val_epoch(loader, model: TrainableModel, tokenizer):
    loss_epoch = 0
    total = 0
    model.eval()

    for iteration, batch in enumerate(tqdm(loader, leave=False)):
        loss = model.step(batch)
        total += 1

        loss_epoch += loss.item()

    return loss_epoch / total
 """

def parse_args(args=None):
    return simple_parsing.parse(Config, args=args)


if __name__ == "__main__":
    import sys

    args = parse_args()
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()

