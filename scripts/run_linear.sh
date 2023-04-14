


# python3 -m torch.distributed.launch --nproc_per_node=8 eval_linear.py \
# --arch resnet50 \
# --lr 0.001 \
# --batch-size-per-gpu 256 \
# --pretrained-weights /data/code/ssl/checkpoints/ssl_ckpt/model_zoo/dino_fcl8_resnet50_pretrain_in1k.crop8.l234.weight.ls0.25.lr0.1.bs2048.ep800_0799_ckpt.pth \
# --checkpoint-key teacher \
# --num_workers 10 \
# --data-path s3://sky/datasets/imagenet/imagenet \
# --output_dir /data/code/ssl/checkpoints/ssl_ckpt/ \
# --method scfs --experiment in1k.l234.crop8.weight.ls0.25.bs2048.ep800.69.8_lr0.001

python3 -m torch.distributed.launch --nproc_per_node=8 eval_linear.py \
--arch resnet50 \
--lr 0.01 \
--batch_size_per_gpu 256 \
--num_workers 10 \
--pretrained_weights /path to pretrained checkpoints/xxx.pth \
--checkpoint_key teacher \
--data_path /path to imagenet/ \
--output_dir output/ \
--method scfs

