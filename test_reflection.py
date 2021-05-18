from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from model import Decomposition_Net_Translation_arbitraryFrameNum
from model import ImageReconstruction_reflection_arbitraryFrameNum_large_FBconcat_AvgMeanPool as ImageReconstruction_reflection_arbitraryFrameNum
from warp_utils import dense_image_warp
import cv2

FLAGS = tf.app.flags.FLAGS

# Define necessary FLAGS
tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_string('test_dataset_name', 'reflection_real_dataset/00001',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('img_type', 'png',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_float('test_ratio', 1.0, """Directory where to write event logs""")
tf.app.flags.DEFINE_string('ckpt_path', None,
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_string('output_dir', 'output',
                           """Output folder.""")

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.insert(1, 'tfoptflow/tfoptflow/')
from copy import deepcopy
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = 'tfoptflow/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'
nn_opts['batch_size'] = 1
nn_opts['gpu_devices'] = ['/device:CPU:0']
nn_opts['controller'] = '/device:CPU:0'
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2
FRAME_NUM = 5


I0 = cv2.imread(FLAGS.test_dataset_name+'_I0.'+FLAGS.img_type).astype(np.float32)[..., ::-1] / 255.0
ORIGINAL_H = I0.shape[0]
ORIGINAL_W = I0.shape[1]

RESIZED_H = int(np.ceil(float(ORIGINAL_H) * FLAGS.test_ratio / 16.0))*16
RESIZED_W = int(np.ceil(float(ORIGINAL_W) * FLAGS.test_ratio / 16.0))*16
print(RESIZED_H)
print(RESIZED_W)


CROP_PATCH_H = RESIZED_H
CROP_PATCH_W = RESIZED_W

def flow_to_img(flow):
    flow_magnitude = tf.sqrt(1e-6 + flow[..., 0]**2.0 + flow[..., 1]**2.0)
    flow_angle = tf.atan2(flow[..., 0], flow[..., 1])

    hsv_0 = ((flow_angle / np.pi)+1.0)/2.0
    hsv_1 = (flow_magnitude - tf.reduce_min(flow_magnitude, axis=[1, 2], keepdims=True)) / (1e-6 + tf.reduce_max(flow_magnitude, axis=[1, 2], keepdims=True) - tf.reduce_min(flow_magnitude, axis=[1, 2], keepdims=True))
    hsv_2 = tf.ones(tf.shape(hsv_0))
    hsv = tf.stack([hsv_0, hsv_1, hsv_2], -1)
    rgb = tf.image.hsv_to_rgb(hsv)

    return rgb

def warp(I, F):
    return tf.reshape(dense_image_warp(I, tf.stack([-F[..., 1], -F[..., 0]], -1)), [FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, 3])

with tf.Graph().as_default():

    fused_frames = [tf.placeholder(tf.float32, [1, CROP_PATCH_H, CROP_PATCH_W, 3]) for _ in range(FRAME_NUM)]
    
    def PWC_full(F, B, lvl_h, lvl_w, pwc_h, pwc_w, lvl, frameNum=FRAME_NUM):
        ratio_h = float(lvl_h) / float(pwc_h)
        ratio_w = float(lvl_w) / float(pwc_w)
        nn = ModelPWCNet(mode='test', options=nn_opts)
        nn.print_config()
        for i in range(frameNum):
            F[i] = tf.image.resize_bilinear(F[i], (pwc_h, pwc_w), align_corners=True)
            B[i] = tf.image.resize_bilinear(B[i], (pwc_h, pwc_w), align_corners=True)

        tmp_list = []
        for i in range(frameNum):
            for j in range(frameNum):
                tmp_list.append(tf.stack([F[i], F[j]], 1))
        for i in range(frameNum):
            for j in range(frameNum):
                tmp_list.append(tf.stack([B[i], B[j]], 1))

        PWC_input = tf.concat(tmp_list, 0)  # [batch_size*20, 2, H, W, 3]
        PWC_input = tf.reshape(PWC_input, [FLAGS.batch_size * (frameNum*frameNum*2), 2, pwc_h, pwc_w, 3])
        pred_labels, _ = nn.nn(PWC_input, reuse=tf.AUTO_REUSE)
        print(pred_labels)

        pred_labels = tf.image.resize_bilinear(pred_labels, (lvl_h, lvl_w), align_corners=True)
        """
        0: W
        1: H
        """
        ratio_tensor = tf.expand_dims(tf.expand_dims(
            tf.expand_dims(tf.convert_to_tensor(np.asarray([ratio_w, ratio_h]), dtype=tf.float32), 0), 0), 0)

        FF = []
        FB = []
        counter = 0
        for i in range(frameNum):
            FF_tmp = []
            FB_tmp = []
            for j in range(frameNum):
                FF_tmp.append(tf.stop_gradient(pred_labels[FLAGS.batch_size * counter:FLAGS.batch_size * (counter+1)] * ratio_tensor))
                FB_tmp.append(tf.stop_gradient(pred_labels[FLAGS.batch_size * (counter+frameNum*frameNum):FLAGS.batch_size * (counter + 1 + frameNum*frameNum)] * ratio_tensor))
                counter += 1
            FF.append(FF_tmp)
            FB.append(FB_tmp)

        return FF, FB


    model = Decomposition_Net_Translation_arbitraryFrameNum(CROP_PATCH_H // 16, CROP_PATCH_W // 16, False, False)
    FF_init, FB_init = model.inference(fused_frames)

    """image"""
    model4 = ImageReconstruction_reflection_arbitraryFrameNum(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=4)
    F_pred_4, B_pred_4 = model4._build_model(fused_frames,
                                             None, None,
                                             FF_init, FB_init)
    FF_3, FB_3 = PWC_full(F_pred_4, B_pred_4,
                          CROP_PATCH_H // (2 ** 4), CROP_PATCH_W // (2 ** 4),
                          int(np.ceil(float(CROP_PATCH_H // (2 ** 4)) / 64.0)) * 64,
                          int(np.ceil(float(CROP_PATCH_W // (2 ** 4)) / 64.0)) * 64, 3)

    model3 = ImageReconstruction_reflection_arbitraryFrameNum(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=3)
    F_pred_3, B_pred_3 = model3._build_model(fused_frames,
                                             F_pred_4, B_pred_4, FF_3, FB_3)

    FF_2, FB_2 = PWC_full(F_pred_3, B_pred_3,
                          CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3),
                          int(np.ceil(float(CROP_PATCH_H // (2 ** 3)) / 64.0)) * 64,
                          int(np.ceil(float(CROP_PATCH_W // (2 ** 3)) / 64.0)) * 64, 2)

    model2 = ImageReconstruction_reflection_arbitraryFrameNum(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=2)
    F_pred_2, B_pred_2 = model2._build_model(fused_frames,
                                             F_pred_3, B_pred_3, FF_2, FB_2)
    FF_1, FB_1 = PWC_full(F_pred_2, B_pred_2,
                          CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2),
                          int(np.ceil(float(CROP_PATCH_H // (2 ** 2)) / 64.0)) * 64,
                          int(np.ceil(float(CROP_PATCH_W // (2 ** 2)) / 64.0)) * 64, 1)

    model1 = ImageReconstruction_reflection_arbitraryFrameNum(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=1)
    F_pred_1, B_pred_1 = model1._build_model(fused_frames,
                                             F_pred_2, B_pred_2, FF_1, FB_1)
    FF_0, FB_0 = PWC_full(F_pred_1, B_pred_1,
                          CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1),
                          int(np.ceil(float(CROP_PATCH_H // (2 ** 1)) / 64.0)) * 64,
                          int(np.ceil(float(CROP_PATCH_W // (2 ** 1)) / 64.0)) * 64, 0)

    model0 = ImageReconstruction_reflection_arbitraryFrameNum(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=0)
    F_pred_0, B_pred_0 = model0._build_model(fused_frames,
                                             F_pred_1, B_pred_1, FF_0, FB_0)

    sess = tf.Session()

    saver2 = tf.train.Saver(var_list=[v for v in tf.all_variables() if "pwcnet" in v.name])
    saver2.restore(sess, nn_opts['ckpt_path'])
    saver4 = tf.train.Saver(var_list=[v for v in tf.all_variables() if
                                      "FeaturePyramidExtractor" in v.name or "TranslationEstimator" in v.name])
    saver4.restore(sess, 'train_dir_initFlow_Reflection/model.ckpt-239999')
    saver5 = tf.train.Saver(var_list=[v for v in tf.all_variables() if "FusionLayer_" in v.name])
    saver5.restore(sess, FLAGS.ckpt_path)
    
    
    import cv2

    out_path = FLAGS.output_dir + '/'
    
    inputs = []
    for frame_idx in range(FRAME_NUM):
        print(FLAGS.test_dataset_name+'_I'+str(frame_idx)+'.'+FLAGS.img_type)
        inputs.append(np.expand_dims(cv2.resize(cv2.imread(FLAGS.test_dataset_name+'_I'+str(frame_idx)+'.'+FLAGS.img_type).astype(np.float32)[..., ::-1] / 255.0, dsize=(RESIZED_W, RESIZED_H), interpolation=cv2.INTER_CUBIC), 0))
    feed_dictionary = {}
    for k,v in zip(fused_frames, inputs):
        feed_dictionary[k] = v
               
    F, B = sess.run([tf.clip_by_value(F_pred_0, 0.0, 1.0), tf.clip_by_value(B_pred_0, 0.0, 1.0)], feed_dict=feed_dictionary)
    
    F_max = np.max(np.concatenate(F, -1))
    B_max = np.max(np.concatenate(B, -1))
    for i in range(5):
        cv2.imwrite(out_path + FLAGS.test_dataset_name[-5:]+'F'+str(i)+'_norm.png', np.clip(np.round(cv2.resize(F[i][0, :, :, ::-1]/F_max, dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC) * 255.0), 0.0, 255.0).astype(np.uint8))
        cv2.imwrite(out_path + FLAGS.test_dataset_name[-5:]+'B'+str(i)+'_norm.png', np.clip(np.round(cv2.resize(B[i][0, :, :, ::-1]/B_max, dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC) * 255.0), 0.0, 255.0).astype(np.uint8))
        cv2.imwrite(out_path + FLAGS.test_dataset_name[-5:]+'F'+str(i)+'.png', np.clip(np.round(cv2.resize(F[i][0, :, :, ::-1], dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC) * 255.0), 0.0, 255.0).astype(np.uint8))
        cv2.imwrite(out_path + FLAGS.test_dataset_name[-5:]+'B'+str(i)+'.png', np.clip(np.round(cv2.resize(B[i][0, :, :, ::-1], dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC) * 255.0), 0.0, 255.0).astype(np.uint8))
