import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.distributed import setup_distributed, cleanup, is_main_process
from models.world_model import GSWM
from data.dataset import DummyEmbodiedDataset

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    # 1. 初始化 DDP
    is_dist, rank, world_size, gpu = setup_distributed()
    device = torch.device(f'cuda:{gpu}')

    # 2. 准备数据
    # 注意：在 DDP 中，必须使用 DistributedSampler
    dataset = DummyEmbodiedDataset(num_samples=160)
    sampler = DistributedSampler(dataset) if is_dist else None
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None),
        sampler=sampler,
        num_components=4 # 远程服务器通常有多核 CPU
    )

    # 3. 初始化模型
    model = GSWM(device=device).to(device)
    if is_dist:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 4. 训练循环
    model.train()
    for epoch in range(args.epochs):
        if is_dist:
            sampler.set_epoch(epoch)
            
        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            text = batch['text'] # 文本通常保持列表形式
            gt = batch['gt_displacements'].to(device)
            
            optimizer.zero_grad()
            
            # 由于目前模型 forward 暂不支持 batch 文本（简单实现中），这里做个循环或修改 forward
            # 简化演示：只取每批次的第一个
            output = model(images[0:1], text[0])
            
            if output is not None:
                pred = output['pred_displacements']
                
                # 损失计算 (对齐检测到的物体数量与 GT)
                # 实际场景中需要匹配算法 (如 Hungarian Matching)，这里简化处理
                min_n = min(pred.shape[0], gt[0].shape[0])
                loss = criterion(pred[:min_n], gt[0][:min_n])
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        if is_main_process():
            print(f"Epoch {epoch} | Loss: {epoch_loss / len(dataloader):.4f}")
            # 保存模型
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

    if is_dist:
        cleanup()

if __name__ == "__main__":
    train()
