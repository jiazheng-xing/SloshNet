python run.py -c ./exp/kin/5w-5s --query_per_class 5 --shot 5 --way 5  \
 --trans_linear_out_dim 1152  --scratch bp --tasks_per_batch 16 --test_iters 2500 \
 --dataset kinetics --split 3 -lr 0.0001 --method resnet50_darts --img_size 224 --scratch bp \
 --num_gpus 4 --opt sgd --save_freq 2500 --print_freq 1000  --training_iterations 30010 --temp_set 1  --weight_decay 5e-4  --steps 0 6 8 9 --LRS 1 0.5 0.1 0.01 \
 --warmup_epochs 0 --warmup_start_lr 0.00001 -step_iter 1000 