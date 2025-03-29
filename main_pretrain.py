import argparse
import datetime
import glob
import json
import os
import time
import traceback
from copy import copy
from pathlib import Path
import torch.nn as nn
import models_mae
from collections import OrderedDict
import numpy as np


import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import util.misc as misc
import util.viz as viz
import wandb
from engine_pretrain import train_one_epoch
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from util.datasets import build_fmow_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torch.nn.functional as F

def nullable_string(val):
    if not val:
        return None
    return val


def get_args_parser():
    parser = argparse.ArgumentParser("pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help=(
            "Batch size per GPU (effective batch size is batch_size * accum_iter * #"
            " gpus"
        ),
    )
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument(
        "--accum_iter",
        type=int,
        default=1,
        help=(
            "Accumulate gradient iterations (for increasing the effective batch size"
            " under memory constraints)"
        ),
    )
    parser.add_argument(
        "--model",
        default="mae_vit_base_Ours",
        type=str,
        metavar="MODEL",
        help=(
            "The name of the model architecture to train. These are defined in"
            " models_mae/__init__.py"
        ),
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="The size of the square-shaped input image",
    )
    parser.add_argument(
        "--patch_size",
        type=str,
        default=16,
        help=(
            "The size of the square-shaped patches across the image. Must be a divisor"
            " of input_size (input_size % patch_size == 0)"
        ),
    )

    parser.add_argument(
        "--print_level",
        type=int,
        default=1,
        help="Print Level (0->3) - Only for MaskedAutoencoderShuntedViT",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.75,
        help="Masking ratio (percentage of removed patches).",
    )

    parser.add_argument(
        "--attn_name",
        type=str,
        default="scaled_dot_product",
        help=(
            "Attention name to use in transformer block. The following require the"
            " --use_xformers flag: 'linformer', 'orthoformer', 'nystrom',"
            " 'fourier_mix', 'local'"
        ),
        choices=[
            "scaled_dot_product",
            "shunted",
            "linformer",
            "orthoformer",
            "nystrom",
            "fourier_mix",
            "local",
        ],
    )

    parser.add_argument(
        "--use_xformers",
        action="store_true",
        help=(
            "Use xFormers instead of Timm for transformer blocks. Not compatible with"
            " --attn_name=shunted"
        ),
    )
    parser.set_defaults(use_xformers=False)

    parser.add_argument(
        "--ffn_name",
        type=str,
        default="MLP",
        choices=["MLP", "FusedMLP"],
        help="Type of FFN layer to use. Only supported if --use_xformers is also set.",
    )

    parser.add_argument(
        "--spatial_mask",
        action="store_true",
        default=False,
        help=(
            "Whether to mask all channels of a spatial location. Only for indp c model"
        ),
    )
    # arg for loss, default is mae
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        help="Loss function to use",
        choices=[
            "mse",
            "mae",
            "l1",
            "l2",
            "bce",
            "ssim",
            "ms_ssim",
            "mse_ssim",
            "mse_ms_ssim",
        ],
    )

    parser.add_argument(
        "--norm_pix_loss",
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help=(
            "Absolute LR. If None, it is set automatically based on absolute_lr ="
            " base_lr * total_batch_size / 256"
        ),
    )
    parser.add_argument(
        "--blr",
        type=float,
        # This is default used in SatMAE paper for pretraining
        default=0.00005,
        metavar="LR",
        help="Base LR. absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="Lower LR bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=40,
        metavar="N",
        help="Defines the epoch where the Warmup Scheduler reaches its maximum value",
    )

    # Dataset parameters
    parser.add_argument(
        "--train_path",
        default="data.json",
        type=str,
        help="Train dataset path",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="maid_rgb",
        choices=[
            "maid_rgb",
            "euro_sat",
            "naip",
            "coco",
        ],
        help="Whether to use fmow rgb, sentinel, or other dataset.",
    )
    parser.add_argument(
        "--masked_bands",
        type=int,
        nargs="+",
        default=None,
        help="Sequence of band indices to mask (with mean val) in sentinel dataset",
    )
    parser.add_argument(
        "--dropped_bands",
        type=int,
        nargs="+",
        default=None,
        help="Which bands (0 indexed) to drop from sentinel data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Path used for saving trained model checkpoints and logs. If not specified,"
            " the directory name is automatically generated based on model config."
        ),
    )
    parser.add_argument(
        "--output_dir_base",
        type=str,
        default="./out",
        help="Base directory to use for model checkpoints directory",
    )

    parser.add_argument(
        "--val_img_path",
        type=str,
        default="./images/",
        help="Path used for saving trained model checkpoints and logs",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=0,
        help="Device to use for training and testing",
    )
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument(
        "--resume",
        type=nullable_string,
        default=None,
        help="The path to the checkpoint to resume training from.",
    )

    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Defines the epoch number to start training from. Useful when resuming"
            " training."
        ),
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="utk-iccv23",
        help="Wandb entity name, eg: utk-iccv23",
    )

    parser.add_argument(
        "--wandb_project",
        type=nullable_string,
        default=None,
        help="Wandb project name, eg: satmae",
    )

    # https://docs.wandb.ai/guides/runs/resuming
    parser.add_argument(
        "--wandb_id",
        type=nullable_string,
        default=None,
        help="Wandb project id, eg: 83faqrtq",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=24,
        help=(
            "The number of CPU workers to use for the data loader. Generally, this"
            " should be set to the number of CPU threads on your machine."
        ),
    )
    parser.add_argument(
        "--pin_mem",
        default=True,
        help=(
            "Pin CPU memory in DataLoader for more efficient (sometimes) transfer to"
            " GPU."
        ),
    )
    parser.add_argument(
        "--no_pin_mem",
        action="store_false",
        dest="pin_mem",
        help=(
            "Don't pin CPU memory in DataLoader. Could severely slow down training on"
            " some systems and datasets."
        ),
    )
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="Number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", default=os.getenv("LOCAL_RANK", 0), type=int
    )  # prev default was -1
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="URL used to set up distributed training"
    )

    return parser



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


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
def main(args):
    misc.init_distributed_mode(args)

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print("=" * 80)
    print(f"{args}".replace(", ", ",\n"))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    #######################################################################################
    print("=" * 80)
    dataset_train = build_fmow_dataset(is_train=True, args=args)

    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        sampler_train = torch.utils.data.DistributedSampler(  # type: ignore
            dataset_train, num_replicas=num_tasks, rank=misc.get_rank(), shuffle=True
        )
        print(f"Sampler_train = {str(sampler_train)}")
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)  # type: ignore

    data_loader_train = torch.utils.data.DataLoader(  # type: ignore
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    #######################################################################################
    # define the model
    model = models_mae.__dict__[args.model](**vars(args))
    model.to(device)

    model_without_ddp = model
    # print(f"Model = {str(model_without_ddp)}")
    teacher = models_mae.__dict__[args.model](**vars(args))
    teacher.to(device)


    #######################################################################################
    print("=" * 80)
    batch_size_eff = args.batch_size * args.accum_iter * misc.get_world_size()

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % batch_size_eff)

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * batch_size_eff / 256

    print("base lr: %.2e" % (args.lr * 256 / batch_size_eff))
    print("actual lr: %.2e" % args.lr)

    # if has_batchnorms(model):
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    if args.distributed:
   
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

        teacher = torch.nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.gpu], find_unused_parameters=True
        ) 
        teacher_without_ddp = teacher.module
    else:
        model_without_ddp = model
        teacher_without_ddp = teacher
 
    teacher_without_ddp.load_state_dict(model_without_ddp.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False


    #######################################################################################
    print("=" * 80)
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        teacher_without_ddp=teacher_without_ddp,
    )

    print("=" * 80)
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    model_num_params = sum(np.prod(p.size()) for p in model_params)
    print(f"Trainable parameters: {model_num_params}")
    _start_m, _final_m = 0.996, 1.0
    _increment = (_final_m - _start_m) / (len(data_loader_train) * args.epochs * 1.25)
    momentum_scheduler = (_start_m + (_increment*i) for i in range(int(len(data_loader_train) * args.epochs*1.25)+1))
    if args.resume:
        for _ in range(args.start_epoch*len(data_loader_train)):
            next(momentum_scheduler)
    #######################################################################################
    # Set up output directory, checkpointing, and logging
    print("=" * 80)
    if args.attn_name == "shunted":
        patch_size = "+".join([str(x) for x in args.patch_size])
    else:
        patch_size = args.patch_size
    model_name: str = "_".join(
        [
            args.model,
            f"xformers-{args.attn_name}-{args.ffn_name}"
            if args.use_xformers
            else f"{args.attn_name}",
            f"i{args.input_size}-p{patch_size}-mr{args.mask_ratio}",
            f"e{args.epochs}-we{args.warmup_epochs}",
            f"b{args.batch_size}-a{args.accum_iter}",
            f"{args.loss}{'-normpix' if args.norm_pix_loss else ''}",
            f"lr{args.lr}",
            args.dataset_type,
        ]
    )

    if args.output_dir is None:
        args.output_dir = f"out_{model_name}"
    if args.output_dir_base is not None:
        args.output_dir = os.path.join(args.output_dir_base, args.output_dir)

    # finding a new output directory if one already exists with the same name
    if args.resume is None:
        if not args.distributed:
            while os.path.exists(args.output_dir):
                # print out if doesn't have any .pth files
                if len(glob.glob(os.path.join(args.output_dir, "*.pth"))) == 0:
                    print(
                        f"INFO: {args.output_dir} already exists, but contains no .pth"
                        " files. You may want to delete it."
                    )
                number = os.path.basename(args.output_dir).split("+")[-1]
                number = int(number) + 1 if number.isdigit() else 1
                args.output_dir = os.path.join(
                    os.path.dirname(args.output_dir), f"out_{model_name}+{number}"
                )
        else:
            if len(glob.glob(os.path.join(args.output_dir, "*.pth"))) != 0:
                raise ValueError(
                    f"ERROR: {args.output_dir} already exists and contains .pth files."
                    " Checkpoints would be overwritten."
                )

    print(f"Output directory: {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    log_writer = None
    if misc.is_main_process():
        if args.wandb_entity is not None and args.wandb_project is not None:
            # resume: (bool, str, optional) Sets the resuming behavior. Options:
            # `"allow"`, `"must"`, `"never"`, `"auto"` or `None`. Defaults to `None`.
            # Cases:
            # - `None` (default): If the new run has the same ID as a previous run,
            #     this run overwrites that data.
            # - `"auto"` (or `True`): if the preivous run on this machine crashed,
            #     automatically resume it. Otherwise, start a new run.
            # - `"allow"`: if id is set with `init(id="UNIQUE_ID")` or
            #     `WANDB_RUN_ID="UNIQUE_ID"` and it is identical to a previous run,
            #     wandb will automatically resume the run with that id. Otherwise,
            #     wandb will start a new run.
            # - `"never"`: if id is set with `init(id="UNIQUE_ID")` or
            #     `WANDB_RUN_ID="UNIQUE_ID"` and it is identical to a previous run,
            #     wandb will crash.
            # - `"must"`: if id is set with `init(id="UNIQUE_ID")` or
            #     `WANDB_RUN_ID="UNIQUE_ID"` and it is identical to a previous run,
            #     wandb will automatically resume the run with the id. Otherwise
            #     wandb will crash.
            wandb_id = wandb.util.generate_id()
            # if args.resume is None  # type: ignore
            # else args.wandb_id
            # )
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=model_name,
                group=args.model,
                job_type="pretrain",
                resume=None,  # if args.resume is None else "must",
                id=wandb_id,
            )
            wandb_args = copy(args)
            # if args.resume is not None:
            #     wandb_args.start_epoch += 1
            wandb.config.update(wandb_args, allow_val_change=True)
            wandb.config.update(
                {"num_params": model_num_params, "batch_size_eff": batch_size_eff}
            )
            wandb.watch(model)
        else:
            print("INFO: Not using WandB.")

        # Logging
        if args.output_dir is not None:
            # output_dir_tb = os.path.join(args.output_dir, "tensorboard")
            output_dir_tb = os.path.join("./logs")
            log_writer = SummaryWriter(log_dir=output_dir_tb)
            print(f"INFO: Tensorboard log path: {output_dir_tb}")
        else:
            print("INFO: Not logging to tensorboard.")

    #######################################################################################

    corpus = torch.load('knowledge_base.pt').detach().to(device)

    div = Divergence()
    criterion = nn.CosineSimilarity(dim=1).to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print(args.start_epoch)
    # for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            teacher = teacher,
            teacher_without_ddp = teacher_without_ddp,
            corpus = corpus,
            div = div,
            criterion = criterion,
            momentum_scheduler = momentum_scheduler,
            args=args,
        )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        print(f"Train stats: {log_stats}")

        plot_img_data_arr = []
        plot_img_title_arr = []
        plot_img_fname_arr = []

        # if args.output_dir and (epoch % 2 == 0 or epoch + 1 == args.epochs):
        if args.output_dir:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                teacher=teacher,
                teacher_without_ddp=teacher_without_ddp,
            )

            # if args.val_img_path is a directory, then we will plot all images in that directory
            if os.path.isdir(args.val_img_path):
                for val_img_path in glob.glob(os.path.join(args.val_img_path, "*.jpg")):
                    plot_img_fname_i = os.path.basename(val_img_path)
                    plot_img_title_i = (
                        f"{model_name} - epoch {epoch} - {plot_img_fname_i}"
                    )
                    plot_img_data_i = viz.plot_reconstruction(
                        model_without_ddp,
                        val_img_path,
                        mask_seed=1234,
                        title=plot_img_title_i,
                        use_noise=None,
                        save=True,
                        savedir=os.path.join(args.output_dir, "plots"),
                        show=False,
                        device=device,
                    )
                    plot_img_fname_arr.append(plot_img_fname_i)
                    plot_img_title_arr.append(plot_img_title_i)
                    plot_img_data_arr.append(plot_img_data_i)
            else:
                plot_img_fname = os.path.basename(args.val_img_path)
                plot_img_title = f"{model_name} - epoch {epoch} - {plot_img_fname}"
                plot_img_data = viz.plot_reconstruction(
                    model_without_ddp,
                    args.val_img_path,
                    mask_seed=1234,
                    title=plot_img_title,
                    use_noise=None,
                    save=True,
                    savedir=os.path.join(args.output_dir, "plots"),
                    show=False,
                    device=device,
                )
                plot_img_fname_arr.append(plot_img_fname)
                plot_img_title_arr.append(plot_img_title)
                plot_img_data_arr.append(plot_img_data)

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.jsonl"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

            # Log all stats from MetricLogger
            try:
                if args.wandb_project is not None:
                    for plot_img_fname, plot_img_title, plot_img_data in zip(
                        plot_img_fname_arr, plot_img_title_arr, plot_img_data_arr
                    ):
                        # For comparison between all models
                        log_stats[f"val_plot_{plot_img_fname}"] = wandb.Image(
                            data_or_path=plot_img_data, caption=plot_img_title
                        )

                    wandb.log(log_stats)
            except ValueError as e:
                traceback.print_exc()
                print(f"Failed to log to wandb: {e}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
