import os
import numpy as np
import pandas as pd
import wave
import librosa
from python_speech_features import *
import sys
import pickle
sys.path.append('/Users/linlin/Desktop/depression/classfication')

import tensorflow.compat.v1 as tf

import vggish.vggish_input as vggish_input
import vggish.vggish_params as vggish_params
import vggish.vggish_postprocess as vggish_postprocess
import vggish.vggish_slim as vggish_slim

import loupe_keras as lpk

from allennlp.commands.elmo import ElmoEmbedder

tf.enable_eager_execution()

elmo = ElmoEmbedder()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))

# Paths to downloaded VGGish files.
checkpoint_path =os.path.join(os.getcwd(),  'vggish/vggish_model.ckpt')
pca_params_path = os.path.join(os.getcwd(), 'vggish/vggish_pca_params.npz')

cluster_size = 16

min_len = 100
max_len = -1

def to_vggish_embedds(x, sr):
    # x为输入的音频，sr为sample_rate
    input_batch = vggish_input.waveform_to_examples(x, sr)
    with tf.Graph().as_default(), tf.Session() as sess:
      vggish_slim.define_vggish_slim()
      vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

      features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
      embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
      [embedding_batch] = sess.run([embedding_tensor],
                                   feed_dict={features_tensor: input_batch})

    # Postprocess the results to produce whitened quantized embeddings.
    pproc = vggish_postprocess.Postprocessor(pca_params_path)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    
    return tf.cast(postprocessed_batch, dtype='float32')

def wav2vlad(wave_data, sr):
    global cluster_size
    signal = wave_data
    melspec = librosa.feature.melspectrogram(signal, n_mels=80,sr=sr).astype(np.float32).T
    melspec = np.log(np.maximum(1e-6, melspec))
    feature_size = melspec.shape[1]
    max_samples = melspec.shape[0]
    output_dim = cluster_size * 16
    feat = lpk.NetVLAD(feature_size=feature_size, max_samples=max_samples, \
                            cluster_size=cluster_size, output_dim=output_dim) \
                                (tf.convert_to_tensor(melspec))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        r = feat.numpy()
    return r
        
def extract_features(number, audio_features, targets, path):
    global max_len, min_len
    if not os.path.exists(os.path.join(prefix, '{1}/{0}/positive_out.wav'.format(number, path))):
        return    
    positive_file = wave.open(os.path.join(prefix, '{1}/{0}/positive_out.wav'.format(number, path)))
    sr1 = positive_file.getframerate()
    nframes1 = positive_file.getnframes()
    wave_data1 = np.frombuffer(positive_file.readframes(nframes1), dtype=np.short).astype(np.float)
    len1 = nframes1 / sr1

    neutral_file = wave.open(os.path.join(prefix, '{1}/{0}/neutral_out.wav'.format(number, path)))
    sr2 = neutral_file.getframerate()
    nframes2 = neutral_file.getnframes()
    wave_data2 = np.frombuffer(neutral_file.readframes(nframes2), dtype=np.short).astype(np.float)
    len2 = nframes2 / sr2

    negative_file = wave.open(os.path.join(prefix, '{1}/{0}/negative_out.wav'.format(number, path)))
    sr3 = negative_file.getframerate()
    nframes3 = negative_file.getnframes()
    wave_data3 = np.frombuffer(negative_file.readframes(nframes3), dtype=np.short).astype(np.float)
    len3 = nframes3/sr3

    for l in [len1, len2, len3]:
        if l > max_len:
            max_len = l
        if l < min_len:
            min_len = l

    with open(os.path.join(prefix, '{1}/{0}/new_label.txt'.format(number, path))) as fli:
        target = float(fli.readline())
    
    if wave_data1.shape[0] < 1:
        wave_data1 = np.array([1e-4]*sr1*5)
    if wave_data2.shape[0] < 1:
        wave_data2 = np.array([1e-4]*sr2*5)
    if wave_data3.shape[0] < 1:
        wave_data3 = np.array([1e-4]*sr3*5)  
    audio_features.append([wav2vlad(wave_data1, sr1), wav2vlad(wave_data2, sr2), \
        wav2vlad(wave_data3, sr3)])
    # targets.append(1 if target >= 53 else 0)
    targets.append(target)


audio_features = []
audio_targets = []

for index in range(114):
    extract_features(index+1, audio_features, audio_targets, 'Data')

for index in range(114):
    extract_features(index+1, audio_features, audio_targets, 'ValidationData')


print("Saving npz file locally...")
np.savez(os.path.join(prefix, 'Features/AudioWhole/whole_samples_reg_%d.npz'%(cluster_size*16)), audio_features)
np.savez(os.path.join(prefix, 'Features/AudioWhole/whole_labels_reg_%d.npz')%(cluster_size*16), audio_targets)

print(max_len, min_len)