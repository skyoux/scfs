#!/usr/bin/env python
import argparse
import os
import sys
import builtins
import datetime
import time
import math
import json
from pathlib import Path
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models as torchvision_models
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from utils import utils
from utils import checkpoint_io
from utils import optimizers
from dataloaders.load_coco import COCOLoader
from augmentations.dino_augmentation import DINODataAugmentation
import backbones.vision_transformer as vits
from models.scfs import *
from utils.box_generator import RandomBoxGenerator

method = "scfs"

def main():
    parser = argparse.ArgumentParser(method, parents=[get_args_parser()])
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    utils.fix_random_seeds(args.seed)
    utils.init_distributed_mode_global(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.multiprocessing_distributed:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(args.gpu, args)

def get_args_parser():
    parser = argparse.ArgumentParser(method, add_help=False)

    #################################
    #### input and output parameters ####
    #################################
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--experiment', default='exp', type=str, help='experiment name')

    #################################
    #### augmentation parameters ####
    #################################
    # multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping.""")
    parser.add_argument('--global_size', type=int, default=224,
        help="""Size of image. Used for large global view cropping.""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate.""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--feature_crops_number', type=int, default=8, help="""Number of small
        local views to generate.""")
    parser.add_argument('--feature_crops_scale', type=float, nargs='+', default=(0.5, 0.8),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    #################################
    ####model parameters ####
    #################################
    parser.add_argument('--arch', default='resnet50', type=str,
        help="""Name of architecture to train. For quick experiments with ViTs""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head..""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    # for ViTs
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16.""")

    #################################
    #### optim parameters ###
    #################################
    # training/pptimization parameters
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    # temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    #################################
    #### dist parameters ###
    #################################
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs.')

    return parser

def train(gpu, args):

    ######################## init dist ########################
    args.gpu = gpu

    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * args.ngpus_per_node + gpu
            print(args.rank)
        dist.init_process_group(backend="nccl", init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    torch.cuda.set_device(args.gpu)
    dist.barrier()

    ######################## preparing data ... ########################
    transform = DINODataAugmentation(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )

    dataset = datasets.ImageFolder(args.data_path, transform=transform)

    # annFile = "/data/code/pretrain/detectron2/datasets/coco/annotations/instances_train2017.json"
    # dataset = datasets.CocoDetection(args.data_path, annFile, transform=transform)
    # dataset = COCOLoader(args.data_path, transform=transform)

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Loaded {len(dataset)} training images.")

    ######################## building networks ...########################
    args.arch = args.arch.replace("deit", "vit")
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = StudentWrapper(
        student, 
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ),
        DINOShallowHead(in_dim=512,
        out_dim=256,
        bottleneck_dim=256,
        hidden_dim=2048
        ),
        DINOShallowHead(in_dim=1024,
        out_dim=256,
        bottleneck_dim=256,
        hidden_dim=2048
        ),
        DINOShallowHead(in_dim=2048,
        out_dim=256,
        bottleneck_dim=256,
        hidden_dim=2048,
        ),
        )
    
    teacher = TeacherWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
        DINOShallowHead(in_dim=512,
            out_dim=256,
            bottleneck_dim=256,
            hidden_dim=2048
        ),
        DINOShallowHead(in_dim=1024,
        out_dim=256,
        bottleneck_dim=256,
        hidden_dim=2048,
        ),
        DINOShallowHead(in_dim=2048,
        out_dim=256,
        bottleneck_dim=256,
        hidden_dim=2048,
        ),
        num_crops=args.local_crops_number)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        # use DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    ######################## preparing loss ... ########################
    dino_loss_fn = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    dino_loss_l2_fn = DINOShallowLoss(
        256,
        args.local_crops_number, 
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    dino_loss_l3_fn = DINOShallowLoss(
        256,
        args.local_crops_number, 
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    dino_loss_l4_fn = DINOShallowLoss(
        256,
        args.local_crops_number, 
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    ######################## preparing optimizer ... ########################
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = optimizers.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    ######################## init schedulers ... ########################
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., 
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    box_generator = RandomBoxGenerator(input_size=args.global_size, 
    min_scale=args.feature_crops_scale[0],
    max_scale=args.feature_crops_scale[1],
    num_patches_per_image = args.feature_crops_number
    )

    summary_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 
    "tb", "{}_{}_pretrain_{}".format(method, args.arch, args.experiment))) if args.rank == 0 else None

    ######################## optionally resume training ########################
    # if finetune
    if args.finetune_checkpoint is not None:
        checkpoint_io.restart_from_checkpoint2(
        args.finetune_checkpoint,
        is_tx_warp=True,
        student=student,
        teacher=teacher,
    )

    to_restore = {"epoch": 0}
    checkpoint_io.restart_from_checkpoint(
        os.path.join(args.output_dir, "{}_{}_pretrain_{}_temp_ckpt.pth".format(method, args.arch, args.experiment)),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss_fn=dino_loss_fn,
        dino_loss_l2_fn=dino_loss_l2_fn,
        dino_loss_l3_fn=dino_loss_l3_fn,
    )
    start_epoch = to_restore["epoch"]

    ######################## start training ########################
    start_time = time.time()
    print("Starting {} training !".format(method))
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        ######################## training one epoch of DINO ... ########################
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, 
            dino_loss_fn, dino_loss_l2_fn, dino_loss_l3_fn, dino_loss_l4_fn,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, summary_writer, box_generator, args)

        ########################writing logs ... ########################
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'arch': args.arch,
            'dino_loss_fn': dino_loss_fn.state_dict(),
            'dino_loss_l2_fn': dino_loss_l2_fn.state_dict(),
            'dino_loss_l3_fn': dino_loss_l3_fn.state_dict(),
            'dino_loss_l4_fn': dino_loss_l4_fn.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        utils.save_on_master(save_dict, os.path.join(args.output_dir, 
        "{}_{}_pretrain_{}_temp_ckpt.pth".format(method, args.arch, args.experiment)))

        if (args.saveckp_freq and epoch % args.saveckp_freq == 0) or epoch == args.epochs - 1:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 
            "{}_{}_pretrain_{}_{:04d}_ckpt.pth".format(method, args.arch, args.experiment, epoch)))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if utils.is_main_process():
            with (Path(args.output_dir) / "{}_{}_pretrain_{}_log.txt".format(method, args.arch, args.experiment)).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    if args.rank == 0:
        summary_writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(student, teacher, teacher_without_ddp, 
                    dino_loss_fn, dino_loss_l2_fn, dino_loss_l3_fn, dino_loss_l4_fn, 
                    data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, summary_writer, box_generator, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    iters_per_epoch = len(data_loader)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate
        it_ = it
        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # student global
            stu_final_g, stu_l2_g, stu_l3_g, stu_l4_g, _, _, _ = student(images[:2])

            # student local
            stu_final_l, stu_l2_l, stu_l3_l, stu_l4_l, stu_f_l2, stu_f_l3, stu_f_l4 = student(images[2:])

            # teacher global
            tea_final_g, tea_l2_g, tea_l3_g, tea_l4_g, tea_l2_l, tea_l3_l, tea_l4_l, _, _, _ = \
                teacher(images[:2], stu_f_l2.detach(), stu_f_l3.detach(), stu_f_l4.detach(), box_generator)

            stu_all = torch.cat([stu_final_g, stu_final_l], dim=0)

            # compulate loss
            loss_final = dino_loss_fn(stu_all, tea_final_g, epoch)

            loss_l2_g, loss_l2_l = dino_loss_l2_fn(stu_l2_g, tea_l2_g, stu_l2_l, tea_l2_l, epoch)
            loss_l3_g, loss_l3_l = dino_loss_l3_fn(stu_l3_g, tea_l3_g, stu_l3_l, tea_l3_l, epoch)
            loss_l4_g, loss_l4_l = dino_loss_l4_fn(stu_l4_g, tea_l4_g, stu_l4_l, tea_l4_l, epoch)
            loss_l2 = (loss_l2_g + loss_l2_l) * 0.5
            loss_l3 = (loss_l3_g + loss_l3_l) * 0.5
            loss_l4 = (loss_l4_g + loss_l4_l) * 0.5

        loss = loss_final + loss_l2 + loss_l3 + loss_l4

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        if utils.is_main_process():
            summary_writer.add_scalar("loss", loss_final.item(),  it)
            summary_writer.add_scalar("lr", optimizer.param_groups[0]["lr"],  it)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    main()