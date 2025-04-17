# 导入标准库
import os
import platform
import argparse
import time
import math
import warnings

# 导入第三方库
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

# 导入 torch 模块
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入自定义模块
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import DPODataset

# 关闭警告信息
warnings.filterwarnings('ignore')

# 日志打印函数（仅主进程打印）
def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

# 学习率调整策略：余弦退火
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

# 将 logits 转换为预测 token 概率（按标签索引）
def logits_to_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)  # logits 转换为 log 概率
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)  # 取标签位置概率
    return probs

# DPO损失函数实现
def dpo_loss(ref_probs, probs, mask, beta):
    # 1. mask.sum 得到每个样本的真实长度，用于后续归一化“序列级别的 log-prob”
    seq_lengths = mask.sum(dim=1, keepdim=True)

    # 2. 按 token 累加 log-probs，再除以长度，得到每个序列的平均 log-prob
    #    ref_probs 与 probs 应该是形状 [batch, seq_len] 的 log π_ref 和 log π_θ
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    probs     = (    probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 3.Batch 里的前一半是“更优”回答 y_w，后一半是“次优”回答 y_ℓ
    batch_size           = ref_probs.shape[0]
    chosen_ref_probs     = ref_probs[:batch_size // 2]   # log π_ref(y_w|x)
    reject_ref_probs     = ref_probs[batch_size // 2:]   # log π_ref(y_ℓ|x)
    chosen_probs         = probs    [:batch_size // 2]   # log π_θ(y_w|x)
    reject_probs         = probs    [batch_size // 2:]   # log π_θ(y_ℓ|x)

    # 4. 计算两个 log-ratio：
    #    A = log π_θ(y_w) - log π_θ(y_ℓ)
    #    B = log π_ref(y_w) - log π_ref(y_ℓ)
    pi_logratios  = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs

    # 5. DPO 核心：用 A – B 作为 sigmoid 的输入
    #    logits = A - B = [log π_θ(y_w) - log π_ref(y_w)]
    #                  - [log π_θ(y_ℓ) - log π_ref(y_ℓ)]
    logits = pi_logratios - ref_logratios

    # 6. 负对数 sigmoid，就是 -log σ(β·logits)
    #    完全对应公式里的 -log σ(β (A–B))
    loss = -F.logsigmoid(beta * logits)

    # 7. 对全batch取平均
    return loss.mean()


# 单个 epoch 的训练流程
def train_epoch(epoch, wandb):
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        # 将 batch 数据移动到指定设备
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)

        # 合并正负样本
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 计算当前学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用 AMP 混合精度训练
        with ctx:
            # 使用参考模型生成 logits（冻结）
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_probs = logits_to_probs(ref_logits, y) * mask

            # 当前模型输出
            outputs = model(x)
            logits = outputs.logits
            probs = logits_to_probs(logits, y) * mask

            # 计算 DPO 损失
            loss = dpo_loss(ref_probs, probs, mask, beta=0.1)
            loss = loss / args.accumulation_steps  # 梯度累积归一

        # 反向传播（使用 AMP Scaler）
        scaler.scale(loss).backward()

        # 梯度累积步长到了才执行更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志打印
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 模型保存
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/rlhf_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()

# 初始化模型和参考模型
def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    # 初始化参考模型（用于计算 log prob）
    ref_model = MiniMindLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)

    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)

    return model, ref_model, tokenizer

# 初始化 DDP 分布式模式
def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind RLHF")
    # 添加各种训练参数
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-RLHF-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/dpo.jsonl")

    args = parser.parse_args()

    # 构造模型配置并准备输出目录
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"
    args.wandb_run_name = f"MiniMind-Full-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置 AMP 模式
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    # 初始化 wandb 日志
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型与 tokenizer
    model, ref_model, tokenizer = init_model(lm_config)

    # 加载 DPO 数据集
    train_ds = DPODataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # AMP 梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # DDP 模型封装
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 启动训练
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
