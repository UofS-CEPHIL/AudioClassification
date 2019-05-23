# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
import os

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim


checkpoint_path = 'vggish_model.ckpt'
pca_params_path = 'vggish_pca_params.npz'
sr = 22050
wav_file = "./data/BabyCryClip/BabyCryClip-0-Z1i1MIWKA.wav"
rel_error = 0.1

input_batch = vggish_input.wavfile_to_examples(wav_file)
print(input_batch.shape)
with tf.Graph().as_default(), tf.Session() as sess:
  vggish_slim.define_vggish_slim()
  vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

  features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
  embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)
  [embedding_batch] = sess.run([embedding_tensor],
                               feed_dict={features_tensor: input_batch})
  print('VGGish embedding: ', embedding_batch[0])

  pproc = vggish_postprocess.Postprocessor(pca_params_path)
  postprocessed_batch = pproc.postprocess(embedding_batch)

  print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
  # print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
