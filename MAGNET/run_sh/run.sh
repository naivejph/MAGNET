#######train
###CUB-cropped
python train.py --model MAGNET --dataset CUB_fewshot_cropped --lr 1e-1 --gamma 1e-1 --disturb_num 5 --weight 0.05 --epoch 400 --stage 3 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 20 --train_shot 5 --train_transform_type 0 --test_shot 1 5 --pre --gpu_num 1
python train.py --model MAGNET --dataset CUB_fewshot_cropped --lr 1e-1 --gamma 1e-1 --disturb_num 1 --weight 0.3  --epoch 400 --stage 3 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 10 --train_shot 5 --train_transform_type 0 --test_shot 1 5 --pre --gpu_num 1 --resnet

######test
###CUB-cropped
python test.py --model MAGNET --disturb_num 5 --weight 0.05 --dataset CUB_fewshot_cropped --model_path "your pth path" --pre
python test.py --model MAGNET --disturb_num 1 --weight 0.3 --dataset CUB_fewshot_cropped --model_path "your pth path" --pre --resnet
