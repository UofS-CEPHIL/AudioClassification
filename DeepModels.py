# -*- coding: utf-8 -*-
import librosa
import numpy as np
import os
import datetime
import pandas as pd
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle
import librosa.display
from pathlib import Path
import pickle
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

features_file_path = "snoring_features.pkl"
labels_file_path = "snoring_labels.pkl"

cnn_features_file_path = "cnn_features.pkl"
cnn_labels_file_path = "cnn_labels.pkl"


class Classification:
    def __init__(self):
        # define global variables path
        self.figure_dir = "./figures/"
        self.data_dir = "./data/"
        self.model_dir = "./models/"
        self.audio_file_ext="*.wav"
        self.tf_record_ext = "*.tfrecords"
        self.vggish_dir = "./features/"

        # vggish configuration
        self.checkpoint_path = 'vggish_model.ckpt'
        self.pca_params_path = 'vggish_pca_params.npz'

        # note: should include other sounds but currently four categories are enough
        self.dir_names = list(["SnoringClip", "SpeechClip"])
        self.n_classes = len(self.dir_names)

        # deep network input
        self.train_data = np.array([])
        self.train_labels = np.array([])
        self.test_data = np.array([])
        self.test_labels = np.array([])
        self.train_size = 0
        self.test_size = 0

        # rnn network configurations
        self.rnn_batch_size = 50
        self.rnn_display_step = 25
        self.rnn_learning_rate = 0.0002  # default learning rate
        self.rnn_training_epoches = 500
        self.rnn_num_hidden = 100
        self.rnn_num_layers = 3
        self.rnn_dropout = 0.5

        # deep network output
        self.train_loss = list()

        # load audio (10 seconds) file names
        self.files, self.labels = list(), list()

        for dir_name in self.dir_names:
            label = self.dir_names.index(dir_name)
            for fn in glob.glob(os.path.join(self.data_dir, dir_name, self.audio_file_ext)):
                self.files.append(fn)
        self.data_size = len(self.files)
        print("Total Audio Data Set: ", self.data_size)

    # audio data features extraction
    # 1. time domain features: sampling
    def extract_features_sampling(self, sampling_rate=100):
        print("extract features(resampling)...")
        samples = list()
        if self.data_size > 0:
            for index in range(self.data_size):
                sound_clip, sample_rate = librosa.load(self.files[index],sr=sampling_rate)
                samples.append(sound_clip)
        return samples  # return (data_size, steps, 1)

    # 2. frequency domain features - mfccs
    def extract_features_mfcc(self, steps=20):
        print("extract features(mfcc)...")
        mfccs = list()
        if self.data_size > 0:
            for index in range(self.data_size):
                sound_clip, sample_rate = librosa.load(self.files[index])
                mfcc = librosa.feature.mfcc(y=sound_clip, sr=sample_rate, n_mfcc=steps)
                mfccs.append(mfcc)
        return mfccs # return (data_size, steps, dim)

    # 3. vggish input as features CNN (cited from google)
    def extract_features_vggish_input(self):
        print("extract features(vggish)...")
        inputs, ilabels = list()
        if self.data_size > 0:
            count = 0
            for file in self.files:
                count += 1
                if count % 100 == 0:
                    print("processing feature # ", count)
                label = self.dir_names.index(file.split("/")[-1].split("-")[0])
                input_batch = vggish_input.wavfile_to_examples(file) # (10, 96, 64)
                if input_batch.shape[0] == 10:
                    inputs.append(input_batch)
                    ilabels.append(label)
        return inputs

    # 4*. Vggish embeddings as features RNN (cited from google)
    def extract_features_vggish_embedding(self):
        """
        :return: features (#, 10, 128); labels (#, 1)
        """
        print("extract features(embeddings)...")
        embeddings, elabels = list(), list()
        if self.data_size > 0:
            count = 0
            for file in self.files:
                count += 1
                if count % 100 == 0:
                    print("processing feature # ", count)
                label = self.dir_names.index(file.split("/")[-1].split("-")[0])
                input_batch = vggish_input.wavfile_to_examples(file) # (10, 96, 64)
                with tf.Graph().as_default(), tf.Session() as sess:
                    vggish_slim.define_vggish_slim()
                    vggish_slim.load_vggish_slim_checkpoint(sess, self.checkpoint_path)

                    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
                    embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
                    [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: input_batch})
                    pproc = vggish_postprocess.Postprocessor(self.pca_params_path)
                    postprocessed_batch = pproc.postprocess(embedding_batch) # (10, 128)
                    # fix length input features

                    if postprocessed_batch.shape[0] == 10:
                        embeddings.append(postprocessed_batch)
                        elabels.append(label)
        print("Total features: ", len(embeddings))

        return embeddings, elabels

    def dump_inputs(self, features, features_file_name, labels, labels_file_name):
        dump_features_file = Path(features_file_name)
        dump_labels_file = Path(labels_file_name)
        if not dump_features_file.is_file() and not dump_labels_file.is_file():
            # files not exist
            with open(features_file_name, 'wb') as file:
                pickle.dump(features, file)
            with open(labels_file_name, 'wb') as file:
                pickle.dump(labels, file)
            return True
        return False

    def load_inputs(self,features_file_name, labels_file_name):
        dump_features_file = Path(features_file_name)
        dump_labels_file = Path(labels_file_name)
        f, l = list(), list()
        if dump_features_file.is_file() and dump_labels_file.is_file():
            # both files exist
            with open(features_file_name, 'rb') as file:
                f = pickle.load(file)
            with open(labels_file_name, 'rb') as file:
                l = pickle.load(file)
        return f, l

    def one_hot_encode(self, labels):
        # convert the specific label as one-hot vector (numpy)
        n_labels = len(labels)
        one_hot_encode = np.zeros((n_labels, self.n_classes))
        one_hot_encode[np.arange(n_labels), labels] = 1.0
        return one_hot_encode

    # before deep learning training testing
    def data_split(self, features, labels):
        # self.n_step, self.n_input = features[0].shape
        print("Total features: ", len(features))
        feature_arr = np.array(features)
        label_arr = np.array(labels)
        if self.data_size > 0 and len(features) == len(labels):
            features_size = len(features)
            ind_list = [i for i in range(len(features))]
            shuffle(ind_list)
            separate_index = int(0.75*features_size)

            self.train_data = np.array([feature_arr[ind_list[i]] for i in range(separate_index)])
            self.train_labels = self.one_hot_encode(np.array([label_arr[ind_list[i]] for i in range(separate_index)]))
            self.test_data = np.array([feature_arr[ind_list[i]] for i in range(separate_index, features_size)])
            self.test_labels = self.one_hot_encode(np.array([label_arr[ind_list[i]] for i in range(separate_index, features_size)]))

            self.train_size = len(self.train_data)
            self.test_size = len(self.test_data)
            print("Training Data Set: ", self.train_size)
            print("Test Data Set: ", self.test_size)

    def rnn_model(self):
        if self.train_size > 0 and self.test_size > 0:
            # define data format
            n_steps, n_channels = self.train_data[0].shape
            tf.reset_default_graph()
            data = tf.placeholder(tf.float32, [None,  n_steps, n_channels])
            target = tf.placeholder(tf.float32, [None, self.n_classes])
            weight = tf.Variable(tf.random_normal([self.rnn_num_hidden, self.n_classes]))
            bias = tf.Variable(tf.random_normal([self.n_classes]))

            # define network structure
            # cell = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_num_hidden, state_is_tuple=True)
            cells = list()
            for _ in range(self.rnn_num_layers):
                cell = tf.nn.rnn_cell.LSTMCell(self.rnn_num_hidden, state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0-self.rnn_dropout)
                cells.append(cell)
            network = tf.nn.rnn_cell.MultiRNNCell(cells)
            val, state = tf.nn.dynamic_rnn(network, data, dtype=tf.float32)
            val = tf.transpose(val, [1, 0, 2])
            last = tf.gather(val, int(val.get_shape()[0]) - 1)

            prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

            # define loss and optimizer
            cross_entropy = -tf.reduce_sum(target*tf.log(prediction))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.rnn_learning_rate).minimize(cross_entropy)

            # evaluate model
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # training model
            init_op = tf.global_variables_initializer()
            self.train_loss = list()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(init_op)
                num_of_batches = int(len(self.train_data)/self.rnn_batch_size)
                for iter in range(self.rnn_training_epoches):
                    ptr = 0
                    for batch in range(num_of_batches):
                        inp, out = self.train_data[ptr:ptr+self.rnn_batch_size], self.train_labels[ptr:ptr+self.rnn_batch_size]
                        ptr += self.rnn_batch_size
                        _, c = sess.run([optimizer,cross_entropy], {data: inp, target: out})
                    train_accu = sess.run(accuracy, {data: self.train_data, target: self.train_labels})
                    self.train_loss.append(1.0 - train_accu)
                    print("epoch - ", str(iter), " training accuracy - ", str(train_accu))
                    # testing
                    if iter % self.rnn_display_step == 0:
                        test_accu = sess.run(accuracy, {data:self.test_data, target: self.test_labels})
                        print("test accuracy - ", str(test_accu))
                # save model
                t = datetime.datetime.now()
                save_path = saver.save(sess, self.model_dir + "rnn_model" + t.strftime("_%Y_%m_%d") + ".ckpt")
                print("Model saved in path: %s" % save_path)

    def plot_learning(self, model_name):
        """
        :param model_name: the classification model name
        """
        if len(self.train_loss) > 0:
            fig = plt.figure()
            plt.title(model_name + " Training Loss")
            ax = fig.add_subplot(111)
            ax.plot(range(len(self.train_loss)), self.train_loss)
            plt.legend()
            plt.xlabel("iterations")
            plt.ylabel("loss")
            t = datetime.datetime.now()
            plt.savefig(model_name + t.strftime("_%Y_%m_%d") + ".png")
            plt.close()


if __name__=='__main__':
    classifier = Classification()
    features, labels = classifier.load_inputs(features_file_path, labels_file_path)
    if len(features) == 0:
        features, labels = classifier.extract_features_vggish_embedding()
        classifier.dump_inputs(features, features_file_path, labels, labels_file_path)
    classifier.data_split(features, labels)
    classifier.rnn_model()
    classifier.plot_learning("RNN")
