import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


# 定义日志记录函数
def Logger(content):
    # 如果不是分布式训练或者是主进程（rank 0），则打印日志
    if not ddp or dist.get_rank() == 0:
        print(content)


# 定义学习率调整函数
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


# 定义训练函数
def train_epoch(epoch, wandb):
    # 交叉熵损失
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 数据移动到设备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 动态调整学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        # 模型有很多层，调整每层学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 根据CPU/GPU选择
        with ctx:
            # 输入X到模型
            res = model(X)
            # 计算Loss
            loss = loss_fct(
                #res.logits.size(-1) 词汇表大小
                res.logits.view(-1, res.logits.size(-1)),# 预测值，形状 (batch_size * sequence_length, vocab_size)
                Y.view(-1) # 形状(batch_size * sequence_length)，底层会做One-Hot处理
            ).view(Y.size()) #形状 (batch_size * sequence_length)

            # *掩码盖住忽略的损失； /掩码用来归一化
            loss = (loss * loss_mask).sum() / loss_mask.sum()

            # 增加模型的辅助损失
            loss += res.aux_loss

            # 将损失除以累积步数
            loss = loss / args.accumulation_steps

        # 梯度缩放避免半精度训练中的数值下溢；之后反向传播
        scaler.scale(loss).backward()
        # 每args.accumulation_steps步更新一次模型参数并情况累计的梯度
        if (step + 1) % args.accumulation_steps == 0:
            # 将梯度还原到正常范围
            scaler.unscale_(optimizer)
            # 对模型参数的梯度进行裁剪，确保梯度的范数（L2 范数）不超过 args.grad_clip 的值。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新模型参数
            scaler.step(optimizer)
            # 更新 GradScaler 的缩放因子。
            scaler.update()
            # 重置优化器的梯度
            optimizer.zero_grad(set_to_none=True)

        # 每过log_interval步，打印当前epoch，loss，lr等日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            # 将训练过程中的一些关键指标（如损失值、学习率和每个 epoch 的时间）记录到 Weights & Biases (wandb) 平台
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 每隔一定步数保存模型
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()  # 切换到评估模式
            moe_path = '_moe' if lm_config.use_moe else ''  # 判断是否使用MoE
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'  # 构建保存路径

            # 获取模型状态字典，判断是否为分布式训练
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)  # 保存模型
            model.train()  # 切换回训练模式


def init_model(lm_config):
    # 加载预训练模型
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config).to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    # 初始化分布式模式
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    # Weights & Biases 日志记录
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    # 加载预训练数据
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    # DistributedSampler确保在多GPU训练时每个进程只处理数据集的一个子集，避免数据重复
    train_sampler = DistributedSampler(train_ds) if ddp else None
    # 创建DataLoader用于批量加载和处理数据

    train_loader =DataLoader(
        train_ds,                   # 数据集对象
        batch_size=args.batch_size, # 每批处理的样本数量
        pin_memory=True,           # 将数据固定在GPU内存中，加速数据转移到GPU的过程
        drop_last=False,           # 保留最后一个不完整批次(如果有)
        shuffle=False,             # 不随机打乱数据，因为当使用DDP时，DistributedSampler已负责数据打乱
        num_workers=args.num_workers, # 用于数据加载的子进程数量
        sampler=train_sampler      # 采样器，在DDP模式下使用DistributedSampler，否则为None
    )
    # 创建梯度缩放器用于混合精度训练，当使用float16或bfloat16时启用
    # 梯度缩放可以防止梯度在半精度计算时下溢出现"梯度消失"问题
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

    # 定义AdamW优化器，相比Adam增加了权重衰减，有助于模型正则化防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        # 在分布式训练中，pos_cis参数不需要进行梯度同步(通常是位置编码相关参数)
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        # 将模型包装为DistributedDataParallel，实现数据并行训练
        # device_ids指定当前进程使用的GPU设备ID
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 计算每个epoch中的迭代次数，用于学习率调整和日志记录
    iter_per_epoch = len(train_loader)

    # 开始训练循环，执行指定epochs次数的训练
    for epoch in range(args.epochs):
        # 调用训练函数处理一个完整epoch的数据
        train_epoch(epoch, wandb)
