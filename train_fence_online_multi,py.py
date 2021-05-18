import os

TRAIN_DIR = 'temp_online_ckpt/'
TRAINING_DATA_PATH = 'fence_imgs/'
TRAINING_SCENE = '00001'
GPU_ID = '0'
IMG_TYPE = 'png'
OUTPUT_DIR = 'output'
OPTIMIZATION_STEPS = 200

# run online optimization

os.system(
    'python36 train_fence_online.py --train_dir ' + TRAIN_DIR + ' --training_data_path ' + TRAINING_DATA_PATH + ' --training_scene ' + TRAINING_SCENE + ' --GPU_ID ' + GPU_ID + ' --max_steps ' + str(
        OPTIMIZATION_STEPS + 10))