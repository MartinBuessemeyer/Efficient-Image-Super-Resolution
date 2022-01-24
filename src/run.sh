# ACTIVE TESTING
# python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_600 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100

# NEXT TIME DO: scale, lr scheduler, batching (better), pruning


# RFDN vs. RFDN ADVANCED BENCHMARKS
#python main.py --model RFDN          --scale 2 --patch_size 256 --save rfdn_600          --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --save_results
#python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --save_results

# PRUNING BENCHMARKS
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_none --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_2 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 2 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_4 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 4 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_8 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 8 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_16 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 16 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_32 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 32 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_64 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 64 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_128 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 128 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_256 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 256 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_512 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 512 --save_results

# PRUNING BENCHMARKS not batchnorm
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_none --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --save_results --disable-batchnorm
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_2 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 2 --save_results --disable-batchnorm
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_4 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 4 --save_results --disable-batchnorm
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_8 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 8 --save_results --disable-batchnorm
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_16 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 16 --save_results --disable-batchnorm
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_32 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 32 --save_results --disable-batchnorm
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_64 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 64 --save_results --disable-batchnorm
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_128 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 128 --save_results --disable-batchnorm
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_256 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 256 --save_results --disable-batchnorm
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_512 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 512 --save_results --disable-batchnorm


# BATCH SIZE BENCHMARKS
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_batch_size_test_1 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 1 --batch_size 1 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_batch_size_test_2 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 2 --batch_size 2 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_batch_size_test_4 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 4 --batch_size 4 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_batch_size_test_8 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 8 --batch_size 8 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_batch_size_test_16 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 16 --batch_size 16 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_batch_size_test_32 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 32 --batch_size 32 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_batch_size_test_64 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 64 --batch_size 64 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_batch_size_test_128 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 128 --batch_size 128 --save_results

# LR SCHEDULER comparison
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_2 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 2 --save_results --lr-scheduler CosineAnnealingWarmRestarts
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_4 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 4 --save_results --lr-scheduler CosineAnnealingWarmRestarts
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_8 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 8 --save_results --lr-scheduler CosineAnnealingWarmRestarts
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_16 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 16 --save_results --lr-scheduler CosineAnnealingWarmRestarts
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_32 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 32 --save_results --lr-scheduler CosineAnnealingWarmRestarts
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_64 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 64 --save_results --lr-scheduler CosineAnnealingWarmRestarts
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_128 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 128 --save_results --lr-scheduler CosineAnnealingWarmRestarts
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_256 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 256 --save_results --lr-scheduler CosineAnnealingWarmRestarts
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_512 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 512 --save_results --lr-scheduler CosineAnnealingWarmRestarts

# LR SCHEDULER Parameters
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_64 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 64 --save_results --lr-scheduler CosineAnnealingWarmRestarts
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_64 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 64 --save_results --lr-scheduler CosineAnnealingWarmRestarts --lr 1e-4 --eta-min 1e-6
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_64 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 64 --save_results --lr-scheduler CosineAnnealingWarmRestarts --lr 1e-3 --eta-min 1e-4
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_64 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 64 --save_results --lr-scheduler CosineAnnealingWarmRestarts --lr 1e-3 --eta-min 1e-5
python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_600_epochs_before_pruning_64 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size 64 --epochs_before_pruning 64 --save_results --lr-scheduler CosineAnnealingWarmRestarts --lr 1e-3 --eta-min 1e-6


# Epochs
#python main.py --model RFDN          --scale 2 --patch_size 256 --save rfdn_1500          --epochs 1500 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --save_results
#python main.py --model RFDN_advanced --scale 2 --patch_size 256 --save rfdn_advanced_1500 --epochs 1500 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --save_results