import os
import builtins
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset, DataLoader, random_split

from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.oss import OSS

import deepspeed

import fire, random, numpy, tqdm

class RndDataset(Dataset):
    # Simple random dataset, replace with your own DS
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        return torch.randn((10000,)), torch.randn((1,))

    def __len__(self):
        return 10000

class Model(nn.Module):
    def __init__(self, features=10000):
        super().__init__()
        self.linear = nn.Linear(features, 5120)
        self.linear2 = nn.Linear(5120, 2560)
        self.linear3 = nn.Linear(2560, 1)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x

class Trainer():
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        accumulate_gradient_batches: int | None = None
    ):
        self.model = model
        self.max_epochs = max_epochs
        self.accumulate_gradient_batches = accumulate_gradient_batches

        self.optimizer = optimizer
        # Use LOCAL_RANK if provided; otherwise fallback to SLURM_LOCALID.
        self.global_rank = int(os.environ['SLURM_PROCID'])
        self.local_rank = self.global_rank % torch.cuda.device_count()
        # Optionally print rank info for debugging:
        if self.global_rank == 0:
            print(f"Global Rank: {self.global_rank}, Local Rank: {self.local_rank}")

    def train_step(self, batch):
        x, y = batch
        y_ = self.model(x)
        loss = nn.functional.mse_loss(y_, y)
        return loss
    
    @torch.no_grad
    def val_step(self, batch):
        x, y = batch
        y_ = self.model(x)
        loss = nn.functional.mse_loss(y_, y)
        return loss
    
    @torch.no_grad
    def test_step(self, batch):
        x, y = batch
        y_ = self.model(x)
        loss = nn.functional.mse_loss(y_, y)
        return loss

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader | None = None):
        for epoch in range(self.max_epochs):
            self.epoch = epoch

            if self.global_rank == 0:
                pbar = tqdm.tqdm(total=len(train_dataloader))
            self.optimizer.zero_grad()
            for batch_idx, batch in enumerate(train_dataloader):
                batch = (x.to(self.local_rank) for x in batch)
                loss = self.train_step(batch)
                if self.accumulate_gradient_batches is not None:
                    loss = loss / self.accumulate_gradient_batches
                loss.backward()

                if self.accumulate_gradient_batches is not None:
                    if (batch_idx + 1) % self.accumulate_gradient_batches == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.global_rank == 0:
                    pbar.set_postfix({"loss": loss.item()})
                    pbar.update(1)

            if self.global_rank == 0:
                pbar.close()

            if isinstance(val_dataloader, DataLoader):
                if self.global_rank == 0:
                    pbar = tqdm.tqdm(total=len(val_dataloader))
                for batch in val_dataloader:
                    batch = (x.to(self.local_rank) for x in batch)
                    loss = self.val_step(batch)
                    if self.global_rank == 0:
                        pbar.set_postfix({"loss": loss.item()})
                        pbar.update(1)
                if self.global_rank == 0:
                    pbar.close()

    def test(self, test_dataloader: DataLoader):
        if self.global_rank == 0:
            pbar = tqdm.tqdm(total=len(test_dataloader))
        for batch in test_dataloader:
            batch = (x.to(self.local_rank) for x in batch)
            loss = self.test_step(batch)
            if self.global_rank == 0:
                pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)
        if self.global_rank == 0:
            pbar.close()

def main(
    batch_size=32,
    lr=1e-4,
    strategy='auto',  # use DDP
    run_name=None,
    resume=None,
    dev=True,  # If dev True, do not connect to WandB (best for development)
    seed=42
):
    # Continue normal setup
    assert strategy.lower() in ['fsdp', 'deepspeed', 'auto', 'ddp', 'fairscale'], "Use FSDP, fsdp, auto, or deepspeed."
    strategy = strategy.lower()

    # The way to do it on a SLURM only system
    # The trick lies in the use of world_size, this tells the distributer how many GPUS in total there are
    # Otherwise it will attempt to spawn all processes on just the GPUs of the master node causing errors
    global_rank = int(os.environ['SLURM_PROCID']) # Which GPU in all gpus
    local_rank = global_rank % torch.cuda.device_count() # Which GPU in the current gpus
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=global_rank
    )

    # Suppress printing if not on master GPU.
    if global_rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    # Tell torch which GPU to use
    torch.cuda.set_device(local_rank)
    
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)

    torch.set_float32_matmul_precision('high')
    # Set device using LOCAL_RANK or SLURM_LOCALID.

    os.makedirs('./checkpoints', exist_ok=True)

    dataset = RndDataset()
    train_size = int(0.7 * len(dataset))
    test_size = int(0.2 * len(dataset))
    val_size = len(dataset) - train_size - test_size

    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Model(features=10000)
    if strategy in ['auto', 'ddp', 'fairscale', 'fsdp']:
        model.cuda(local_rank) # First move the model to the GPU then DDP it
        if strategy in ['auto', 'ddp']:
            model = DDP(model, device_ids=[local_rank]) # Move the model in DDP setting to the GPU
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        elif strategy == 'fsdp':
            model = FSDP(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        elif strategy == 'fairscale':
            optimizer = OSS(params=model.parameters(), optim=torch.optim.Adam, lr=lr)
            model = ShardedDDP(model, optimizer)
    elif strategy == 'deepspeed':
        os.environ['LOCAL_RANK'] = str(local_rank) # For Deepspeed
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model, optimizer, _, _ = deepspeed.initialize(
            model=model, 
            optimizer=optimizer, 
            config="deepspeed_config.json"
        )

    trainer = Trainer(model, optimizer, max_epochs=250)
    trainer.train(train_loader, val_loader)

    dist.destroy_process_group()

if __name__ == "__main__":
    fire.Fire(main)
