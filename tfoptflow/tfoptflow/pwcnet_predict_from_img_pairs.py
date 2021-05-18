"""
pwcnet_predict_from_img_pairs.py

Run inference on a list of images pairs.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
from copy import deepcopy
import numpy as np
import scipy as sp
from PIL import Image
from scipy import misc
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from visualize import display_img_pairs_w_flows
import tensorflow as tf

sess = tf.Session()

# TODO: Set device to use for inference
# Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
gpu_devices = ['/device:GPU:1']
controller = '/device:GPU:1'

# TODO: Set the path to the trained model (make sure you've downloaded it first from http://bit.ly/tfoptflow)
ckpt_path = './models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

# Build a list of image pairs to process
img_pairs = []
# image_path1 = '../../vimeo_septuplet/vimeo_septuplet/sequences/00050/0001/im1.png'
# image_path2 = '../../vimeo_septuplet/vimeo_septuplet/sequences/00050/0001/im5.png'
image_path1 = 'samples/mpisintel_test_clean_ambush_1_frame_0001.png'
image_path2 = 'samples/mpisintel_test_clean_ambush_1_frame_0002.png'
image1, image2 = sp.misc.imread(image_path1), sp.misc.imread(image_path2)
img_pairs.append((image1, image2))

print(np.max(image1))
print(np.min(image1))

# Configure the model for inference, starting with the default options
nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = ckpt_path
nn_opts['batch_size'] = 1
nn_opts['gpu_devices'] = gpu_devices
nn_opts['controller'] = controller

# We're running the PWC-Net-large model in quarter-resolution mode
# That is, with a 6 level pyramid, and upsampling of level 2 by 4 in each dimension as the final flow prediction
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2

# The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
# of 64. Hence, we need to crop the predicted flows to their original size
nn_opts['adapt_info'] = (1, 436, 1024, 2)

# Instantiate the model in inference mode and display the model configuration
nn = ModelPWCNet(mode='test', options=nn_opts)
nn.print_config()

# Generate the predictions and display them
pred_labels = nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)[0]
# display_img_pairs_w_flows(img_pairs, pred_labels)

import tensorflow as tf
import core_warp
import cv2
sess = tf.Session()
img2 = tf.placeholder(tf.float32, [1, 436, 1024, 3])
flo = tf.placeholder(tf.float32, [1, 436, 1024, 2])
# flo = tf.stack([-flo[..., 1], -flo[..., 0]], -1)
out_img1 = core_warp.dense_image_warp(img2, flo)
fff = np.stack((pred_labels[..., 0], pred_labels[..., 1]), -1)
out_img1_np = sess.run(out_img1, feed_dict={img2:np.expand_dims(image2.astype(np.float32)/255.0, 0), flo:np.expand_dims(fff, 0)})
cv2.imwrite('samples/wapred0.png', np.round(out_img1_np[0, :, :, ::-1]*255.0).astype(np.uint8))

fff = np.stack((-pred_labels[..., 0], pred_labels[..., 1]), -1)
out_img1_np = sess.run(out_img1, feed_dict={img2:np.expand_dims(image2.astype(np.float32)/255.0, 0), flo:np.expand_dims(fff, 0)})
cv2.imwrite('samples/wapred1.png', np.round(out_img1_np[0, :, :, ::-1]*255.0).astype(np.uint8))

fff = np.stack((pred_labels[..., 0], -pred_labels[..., 1]), -1)
out_img1_np = sess.run(out_img1, feed_dict={img2:np.expand_dims(image2.astype(np.float32)/255.0, 0), flo:np.expand_dims(fff, 0)})
cv2.imwrite('samples/wapred2.png', np.round(out_img1_np[0, :, :, ::-1]*255.0).astype(np.uint8))

fff = np.stack((-pred_labels[..., 0], -pred_labels[..., 1]), -1)
out_img1_np = sess.run(out_img1, feed_dict={img2:np.expand_dims(image2.astype(np.float32)/255.0, 0), flo:np.expand_dims(fff, 0)})
cv2.imwrite('samples/wapred3.png', np.round(out_img1_np[0, :, :, ::-1]*255.0).astype(np.uint8))

fff = np.stack((pred_labels[..., 1], pred_labels[..., 0]), -1)
out_img1_np = sess.run(out_img1, feed_dict={img2:np.expand_dims(image2.astype(np.float32)/255.0, 0), flo:np.expand_dims(fff, 0)})
cv2.imwrite('samples/wapred4.png', np.round(out_img1_np[0, :, :, ::-1]*255.0).astype(np.uint8))

fff = np.stack((-pred_labels[..., 1], pred_labels[..., 0]), -1)
out_img1_np = sess.run(out_img1, feed_dict={img2:np.expand_dims(image2.astype(np.float32)/255.0, 0), flo:np.expand_dims(fff, 0)})
cv2.imwrite('samples/wapred5.png', np.round(out_img1_np[0, :, :, ::-1]*255.0).astype(np.uint8))

fff = np.stack((pred_labels[..., 1], -pred_labels[..., 0]), -1)
out_img1_np = sess.run(out_img1, feed_dict={img2:np.expand_dims(image2.astype(np.float32)/255.0, 0), flo:np.expand_dims(fff, 0)})
cv2.imwrite('samples/wapred6.png', np.round(out_img1_np[0, :, :, ::-1]*255.0).astype(np.uint8))

fff = np.stack((-pred_labels[..., 1], -pred_labels[..., 0]), -1)
out_img1_np = sess.run(out_img1, feed_dict={img2:np.expand_dims(image2.astype(np.float32)/255.0, 0), flo:np.expand_dims(fff, 0)})
cv2.imwrite('samples/wapred7.png', np.round(out_img1_np[0, :, :, ::-1]*255.0).astype(np.uint8))



# image1_np = image1.astype(np.float32) / 255.0
# image2_np = image2.astype(np.float32) / 255.0
# inin_np = np.stack([image1_np, image2_np], 0)
# inin_np = np.expand_dims(inin_np, 0)
# print(image1_np.shape)
# inin = tf.placeholder(tf.float32, [1, 2, 256, 448, 3])
# outout = nn.nn(inin)
# out_np = sess.run(outout, feed_dict={inin:inin_np})
# display_img_pairs_w_flows(img_pairs, out_np)

