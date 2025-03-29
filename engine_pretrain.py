# --------------------------------------------------------
# References:
# SatMAE: https://github.com/sustainlab-group/SatMAE
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import contextlib
import math
import sys
from typing import Iterable

import torch

import util.lr_sched as lr_sched
import util.misc as misc
import wandb
import torch.nn.functional as F
import torch.nn as nn
class Divergence(torch.nn.Module):
    """
    Jensen-Shannon divergence
    """

    def __init__(self):
        super(Divergence, self).__init__()
        self.eps = 1e-7  # Small constant for numerical stability

    def forward(self, p: torch.Tensor, q: torch.Tensor):
        # Ensure p and q are probability distributions
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)

        # Calculate the mixture distribution M
        m = 0.5 * (p + q)

        # Add a small constant to avoid log(0)
        m = m.clamp(min=self.eps)

        # Calculate KL divergence from p to m and from q to m
        kl_p_m = F.kl_div(p.log(), m, reduction='batchmean')
        kl_q_m = F.kl_div(q.log(), m, reduction='batchmean')

        # Jensen-Shannon divergence
        js_div = 0.5 * (kl_p_m + kl_q_m)

        return js_div


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    x_rank = x.argsort(dim=1)
    ranks = torch.zeros_like(x_rank, dtype=torch.float)
    n, d = x_rank.size()

    for i in range(n):
        ranks[i][x_rank[i]] = torch.arange(d, dtype=torch.float).to(ranks.device)
    return ranks

def cal_spr_corr(x: torch.Tensor, y: torch.Tensor):
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    x_rank_mean = torch.mean(x_rank, dim=1).unsqueeze(1)
    y_rank_mean = torch.mean(y_rank, dim=1).unsqueeze(1)
    xn = x_rank - x_rank_mean
    yn = y_rank - y_rank_mean
    x_var = torch.sqrt(torch.sum(torch.square(xn), dim=1).unsqueeze(1))
    y_var = torch.sqrt(torch.sum(torch.square(yn), dim=1).unsqueeze(1))
    xn = xn / x_var
    yn = yn / y_var

    return torch.mm(xn, torch.transpose(yn, 0, 1))
def return_sim_matrix(image_feature_1, image_feature_2, temp=0.05):
    sim_matrix = F.cosine_similarity(image_feature_1.unsqueeze(0), image_feature_2.unsqueeze(1), dim=-1)
    sim_matrix = sim_matrix / temp
    return sim_matrix



def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    teacher=None,
    teacher_without_ddp=None,
    corpus=None,
    div=None,
    criterion=None,
    momentum_scheduler = None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    for data_iter_step, samples in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)

        # with torch.cuda.amp.autocast():
        loss, p1, p2, _, _= model(samples, mask_ratio=args.mask_ratio)
        # loss, p2, _= model(samples, mask_ratio=args.mask_ratio)
        with torch.no_grad():
            t_p1,_ = teacher(samples, mask_ratio=args.mask_ratio, is_teacher=True)


        sim_matrix =return_sim_matrix(p1, t_p1, 0.2)
        rank_1 = F.cosine_similarity(p1.unsqueeze(1), corpus.unsqueeze(0), dim=-1)
        rank_2 = F.cosine_similarity(t_p1.unsqueeze(1), corpus.unsqueeze(0), dim=-1)
        cos_sim_baseE = cal_spr_corr(rank_1, rank_2)
        cos_sim_baseE = cos_sim_baseE.cuda()
        loss_fct_baseE = nn.MSELoss(reduction="none")
        cos_sim_baseE_bound = torch.logical_and(cos_sim_baseE <= 1.1, cos_sim_baseE >= -1.1).type(torch.float).cuda()
        mse = loss_fct_baseE(sim_matrix, cos_sim_baseE)
        loss_baseE1 = torch.sum(mse * cos_sim_baseE_bound) / (torch.sum(cos_sim_baseE_bound) + 1e-8) 

        sim_matrix =return_sim_matrix(p2, t_p1, 0.2)
        rank_1 = F.cosine_similarity(p2.unsqueeze(1), corpus.unsqueeze(0), dim=-1)
        rank_2 = F.cosine_similarity(t_p1.unsqueeze(1), corpus.unsqueeze(0), dim=-1)
        cos_sim_baseE = cal_spr_corr(rank_1, rank_2)
        cos_sim_baseE = cos_sim_baseE.cuda()
        loss_fct_baseE = nn.MSELoss(reduction="none")
        cos_sim_baseE_bound = torch.logical_and(cos_sim_baseE <= 1.1, cos_sim_baseE >= -1.1).type(torch.float).cuda()
        mse = loss_fct_baseE(sim_matrix, cos_sim_baseE)
        loss_baseE2 = torch.sum(mse * cos_sim_baseE_bound) / (torch.sum(cos_sim_baseE_bound) + 1e-8)
        loss_baseE = loss_baseE1/2.0 + loss_baseE2/2.0

        sim_matrix_1 = return_sim_matrix(p1, t_p1, 0.05)
        sim_matrix_2 = return_sim_matrix(t_p1, p1, 0.05)
        sd_loss1 = div(sim_matrix_1, sim_matrix_2)

        sim_matrix_1 = return_sim_matrix(p2, t_p1, 0.05)
        sim_matrix_2 = return_sim_matrix(t_p1, p2, 0.05)
        sd_loss2 = div(sim_matrix_1, sim_matrix_2)
        sd_loss = sd_loss1/2.0 + sd_loss2/2.0

        loss_cd = -(criterion(p1, t_p1).mean() + criterion(p2, t_p1).mean()) * 0.5

        loss_cd = 2 * loss_cd + 0.05 * loss_baseE + sd_loss

        loss = loss  + loss_cd

        loss_value = loss.item()

        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")


            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
           
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar("train_loss", train_loss_step, epoch_1000x)
                log_writer.add_scalar("lr", train_lr_step, epoch_1000x)

                # Wandb logging
                if args.local_rank == 0 and args.wandb_project is not None:
                    with contextlib.suppress(ValueError):
                        wandb.log(
                            {
                                "train_loss_step": train_loss_step,
                                "train_lr_step": train_lr_step,
                                "epoch_1000x": epoch_1000x,
                            }
                        )
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

            # raise ValueError(f"Loss is {loss_value}, stopping training")
            
            # sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # EMA update for the teacher
        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(model.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        
        metric_logger.update(loss=loss_value)

        train_lr_step = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=train_lr_step)

        train_loss_step = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
           
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", train_loss_step, epoch_1000x)
            log_writer.add_scalar("lr", train_lr_step, epoch_1000x)

            # Wandb logging
            if args.local_rank == 0 and args.wandb_project is not None:
                with contextlib.suppress(ValueError):
                    wandb.log(
                        {
                            "train_loss_step": train_loss_step,
                            "train_lr_step": train_lr_step,
                            "epoch_1000x": epoch_1000x,
                        }
                    )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
