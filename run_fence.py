import os

INIT_DIR = 'train_dir_initFlow_Fence/'
TRAIN_DIR = 'temp_online_ckpt/'
TRAINING_DATA_PATH = 'fence_imgs/'
TRAINING_SCENE = '00006'
GPU_ID = '0'
IMG_TYPE = 'png'
OUTPUT_DIR = 'output'
OPTIMIZATION_STEPS = 200

# run online optimization
os.system(
    'python3 train_fence_online.py --train_dir ' + TRAIN_DIR + ' --training_data_path ' + TRAINING_DATA_PATH  + ' --GPU_ID ' + GPU_ID + ' --max_steps ' + str(
        OPTIMIZATION_STEPS + 10))

# inference with the optimized weights
os.system(
    'python3 test_fence.py --test_dataset_name ' + TRAINING_DATA_PATH + '/' + TRAINING_SCENE + ' --img_type ' + IMG_TYPE + ' --ckpt_path ' + TRAIN_DIR + 'model.ckpt-' + str(
        OPTIMIZATION_STEPS) + ' --output_dir ' + OUTPUT_DIR)

# os.system(
#     'python3 test_fence.py --test_dataset_name ' + TRAINING_DATA_PATH + '/' + TRAINING_SCENE + ' --img_type ' + IMG_TYPE + ' --ckpt_path ' + INIT_DIR + 'model.ckpt-239999' + ' --output_dir ' + OUTPUT_DIR)
