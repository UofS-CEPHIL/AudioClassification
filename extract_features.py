import librosa.display
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import tensorflow as tf
figure_dir = "./figures/"
data_dir = "./data/"
dir_names = list(["CoughClip", "HiccupClip", "SneezeClip", "SnoringClip", "BabyCryClip"])
audio_file_ext="*.wav"
tf_record_ext = "*.tfrecords"
vggish_dir = "./features/"


def dump_vggish_features():
    for dir_name in dir_names:
        count = 0
        label = dir_names.index(dir_name)
        for file in glob.glob(os.path.join(data_dir, dir_name, audio_file_ext)):
            command = "python ./VggishFeatures/vggish_inference_demo.py"
            param1 = " --wav_file {0}".format(file)
            store_tfrecords_path = vggish_dir + dir_name + "/" + file.split("/")[-1].split(".")[0] + ".tfrecords"
            print(store_tfrecords_path)
            param2 = " --tfrecord_file {0}".format(store_tfrecords_path)
            param3 = " --pca_params ./VggishFeatures/vggish_pca_params.npz"
            param4 = " --checkpoint ./VggishFeatures/vggish_model.ckpt"
            os.system(command+param1+param2+param3+param4)
            count += 1
            if count > 1:
                break

def load_vggish_features():
    for dir_name in dir_names:
        label = dir_names.index(dir_name)
        files = list()
        count = 0
        for file in glob.glob(os.path.join(vggish_dir, dir_name, tf_record_ext)):
            # create file queue
            files.append(file)
            record_iterator = tf.python_io.tf_record_iterator(path=file)

            for string_record in record_iterator:
                example = tf.train.SequenceExample()
                example.ParseFromString(string_record)
                # 10 seconds -> 10 steps for each step 128 d
                print(example.feature_lists.feature_list["audio_embedding"].feature[0].bytes_list.value)
            count += 1
            if count >= 1:
                break
        break


dump_vggish_features()
load_vggish_features()







def plot_spectrum_features():
    for dir_name in dir_names:
        count = 0
        label = dir_names.index(dir_name)
        for fn in glob.glob(os.path.join(data_dir, dir_name, audio_file_ext)):
            sound_file, sr = librosa.load(fn)
            plt.figure(figsize=(12, 8))
            # wave plot
            plt.subplot(4, 2, 1)
            librosa.display.waveplot(sound_file, sr=sr)
            plt.title('Wave spectrogram (Amplitude)')

            # feature extraction
            # STFT power spectrum
            plt.subplot(4, 2, 2)
            STFT = librosa.amplitude_to_db(librosa.stft(sound_file), ref=np.max)  # db fen bei
            print(STFT.shape)
            librosa.display.specshow(STFT, y_axis='log')   #or y_axis= linear; log
            plt.colorbar(format='%+2.0f dB')
            plt.title('Log-frequency power spectrogram')

            # CQT
            plt.subplot(4, 2, 3)
            CQT = librosa.amplitude_to_db(librosa.cqt(sound_file, sr=sr), ref=np.max)
            print(CQT.shape)
            librosa.display.specshow(CQT, y_axis='cqt_hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Constant-Q power spectrogram (Hz)')

            # tempogram
            plt.subplot(4, 2, 4)
            Tgram = librosa.feature.tempogram(y=sound_file, sr=sr)
            librosa.display.specshow(Tgram, x_axis='time', y_axis='tempo')
            plt.colorbar()
            plt.title('Tempogram')
            plt.tight_layout()

            plt.subplot(4, 2, 5)
            MFCCS = librosa.feature.mfcc(y=sound_file, sr=sr, n_mfcc=20)
            print(MFCCS.T.flatten().shape)
            librosa.display.specshow(MFCCS, x_axis='time')
            plt.colorbar()
            plt.title('MFCC')
            plt.tight_layout()

            plt.subplot(4, 2, 6)
            ROLLOFF = librosa.feature.spectral_rolloff(y=sound_file, sr=sr)
            plt.semilogy(ROLLOFF.T, label='Roll-off frequency')
            plt.ylabel('Hz')
            plt.xticks([])
            plt.xlim([0, ROLLOFF.shape[-1]])
            plt.title('spectral rolloff')

            plt.subplot(4, 2, 7)
            Cent = librosa.feature.spectral_centroid(y=sound_file, sr=sr)
            plt.semilogy(Cent.T, label='Spectral centroid')
            plt.ylabel('Hz')
            plt.xticks([])
            plt.xlim([0, Cent.shape[-1]])
            plt.legend()

            plt.subplot(4, 2, 8)
            Mel = librosa.feature.melspectrogram(y=sound_file, sr=sr, n_mels=128,fmax=8000)
            librosa.display.specshow(librosa.power_to_db(Mel,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel spectrogram')
            plt.tight_layout()

            store_full_path = figure_dir + dir_name + "/" + fn.split("/")[-1].split(".")[0]+".png"
            print("Store position: ", store_full_path)
            plt.savefig(store_full_path)

            count += 1
            if count > 1:
                break
