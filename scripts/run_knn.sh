
# knn eval

# dino_fcl7_resnet50_pretrain_in1k_crop8_4_l234_lr0.1_0170_ckpt.pth
# dino_fcl8_resnet50_pretrain_in1k.crop8.l234.weight.ls0.25.lr0.1.bs256.ep200_temp_ckpt.pth
# dino_fcl8_resnet50_pretrain_in1k.crop8.l234.weight.ls0.25.lr0.1.bs1024.ep800_0300_ckpt.pth
# dino_fcl8_resnet50_pretrain_in1k.crop8.l234.weight.ls0.25.lr0.1.bs2048.ep800_0650_ckpt.pth

### pretrain on coco

# dino_resnet50_pretrain_coco.crop8.bs256.ep200.lr0.3_0199_ckpt.pth
# dino_fcl8_resnet50_pretrain_coco.crop8.l234.weight.ls0.25.bs256.ep200.lr0.3_0199_ckpt.pth

python3 -m torch.distributed.launch --nproc_per_node=1 eval_knn.py \
--arch resnet50 \
--batch_size_per_gpu 512 \
--pretrained_weights /data/code/ssl/checkpoints/ssl_ckpt/dino_fcl8_resnet50_pretrain_coco.crop8.l234.weight.ls0.25.bs256.ep200.lr0.3_0199_ckpt.pth \
--checkpoint_key teacher \
--num_workers 20 \
--data_path s3://sky/datasets/imagenet/imagenet \
--use_cuda False \
--method scfs

python3 -m torch.distributed.launch --nproc_per_node=8 eval_knn.py \
--arch resnet18 \
--batch_size_per_gpu 512 \
--pretrained_weights /path to pretrained checkpoints/xxx.pth \
--checkpoint_key teacher \
--num_workers 20 \
--data_path /path to imagenet/ \
--use_cuda True \
--method scfs