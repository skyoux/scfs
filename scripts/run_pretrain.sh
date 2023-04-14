

python3 main_scfs.py \
--arch resnet50 \
--optimizer sgd --lr 0.1 --min_lr 0.0048 \
--weight_decay 1e-4 --weight_decay_end 1e-4 \
--warmup_teacher_temp 0.04 --teacher_temp 0.07 --warmup_teacher_temp_epochs 50 \
--global_crops_scale 0.25 1 --local_crops_scale 0.05 0.25 \
--dist_url 'tcp://localhost:10009' --multiprocessing_distributed --world_size 1 --rank 0 \
--batch_size_per_gpu 32 \
--use_fp16 True \
--data_path s3://sky/datasets/imagenet/imagenet \
--epoch 200 \
--output_dir /data/code/ssl/checkpoints/ssl_ckpt/ \
--experiment in1k.crop8.l234.weight.ls0.25.lr0.1.bs256.ep200 \


python3 main_scfs.py \
--arch resnet50 \
--optimizer sgd --lr 0.1 --min_lr 0.0048 \
--weight_decay 1e-4 --weight_decay_end 1e-4 \
--warmup_teacher_temp 0.04 --teacher_temp 0.07 --warmup_teacher_temp_epochs 50 \
--global_crops_scale 0.25 1 --local_crops_scale 0.05 0.25 \
--dist_url 'tcp://localhost:10009' --multiprocessing_distributed --world_size 1 --rank 0 \
--batch_size_per_gpu 32 \
--use_fp16 True \
--epoch 200 \
--data_path /path to imagenet/ \
--output_dir output/ \
