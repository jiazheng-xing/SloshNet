python run.py -c ./exp/ssv2/5w-1s --query_per_class 1 --shot 1 --way 5  \
 --trans_linear_out_dim 1152  --scratch bp --tasks_per_batch 16 --test_iters 8000 \
 --dataset ssv2 --split 7 -lr 0.001 --method resnet50_darts --img_size 224 --scratch bp \
 --num_gpus 1 --opt sgd --save_freq 8000 --print_freq 1000  --training_iterations 80010 --temp_set 1 2  --weight_decay 5e-5 --steps 0 6 8 9 --LRS 1 0.5 0.1 0.01  \
 --warmup_epochs 0 --warmup_start_lr 0.00001 -step_iter 8000 