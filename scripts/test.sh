python run.py -c ./exp/kin/tst_1 --query_per_class 1 --shot 1 --way 5  \
 --trans_linear_out_dim 1152  --scratch bp --tasks_per_batch 16 --test_iters 2500 \
 --dataset kinetics --split 3 -lr 0.0001 --method resnet50_darts --img_size 224 --scratch bp \
 --num_gpus 1 --opt sgd --save_freq 2500 --print_freq 1000  --training_iterations 30010 --temp_set 1 --test_model_only True --test_model_path ./exp/checkpoint.pt
