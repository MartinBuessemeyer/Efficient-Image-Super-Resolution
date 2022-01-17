# ACTIVE TESTING
# python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_600 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100

# NEXT TIME DO: scale


# RFDN vs. RFDN ADVANCED BENCHMARKS
python main.py --model RFDN          --scale 2 --patch_size 96 --save rfdn_600          --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_600 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --save_results

# PRUNING BENCHMARKS
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_600_epochs_before_pruning_none --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_600_epochs_before_pruning_19 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --epochs_before_pruning 19 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_600_epochs_before_pruning_48 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --epochs_before_pruning 48 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_600_epochs_before_pruning_98 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --epochs_before_pruning 90 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_600_epochs_before_pruning_140 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --epochs_before_pruning 140 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_600_epochs_before_pruning_180 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --epochs_before_pruning 180 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_600_epochs_before_pruning_250 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --epochs_before_pruning 250 --save_results

# BATCH SIZE BENCHMARKS
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_batch_size_test_1 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 1 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_batch_size_test_8 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 8 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_batch_size_test_16 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 16 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_batch_size_test_32 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 32 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_batch_size_test_64 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 64 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_batch_size_test_128 --epochs 600 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --batch_size_test 128 --save_results

# Epochs
python main.py --model RFDN          --scale 2 --patch_size 96 --save rfdn_1500          --epochs 1500 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --save_results
python main.py --model RFDN_advanced --scale 2 --patch_size 96 --save rfdn_advanced_1500 --epochs 1500 --dir_data=/mnt/ssd1/midl21t1/datasets --reset --data_test DIV2K+Set5+BSD100+Urban100 --data_validate DIV2K+Set5+BSD100+Urban100 --save_results