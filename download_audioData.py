from __future__ import unicode_literals
import youtube_dl
import csv
import json
import os
import glob
# some health-related labels:
Speech = "Speech" #/m/09x0r
Cough = "Cough"  #/m/01b_21
BabyCry = "Baby cry, infant cry"
Sneeze = "Sneeze"
Cry = "Crying, sobbing"
Snoring = "Snoring"
Hiccup = "Hiccup"
# url links
balance_dataset_file = "../balanced_train_segments.csv"
eval_dataset_file = "../eval_segments.csv"
unbalanced_dataset_file = "../unbalanced_train_segments.csv"
# all dataset
all_files = list([balance_dataset_file, unbalanced_dataset_file, eval_dataset_file])


# read url for specific label
def read_urls(label_id, files):
    """
    :param label: string
    :return: url lists
    """
    url_list = list()
    for file in files:
        with open(file, 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                for ele in range(3, len(row)):
                    fid = row[ele].replace('"','')
                    fid = fid.replace(' ', '')
                    if fid == label_id:
                        start = int(float(row[1].replace(' ', '')))
                        end = int(float(row[2].replace(' ', '')))
                        url_list.append((row[0], str(start), str(end)))
                        break
    print("Total records: " + str(len(url_list)))
    return url_list


def get_audio_data(urls, path, label):
    """
    :param urls: the link, start point, end point of the audio
    :path: the path where audio data will store
    :label: the class of audio data
    """
    pre_url = "http://youtu.be/"

    ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': path+'/{0}/{0}-%(id)s.%(ext)s'.format(label),
    'ignoreerrors': True,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
    }
    for ele in range(len(urls)):
        if ele > 1500:
            break
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            actual_url = pre_url + urls[ele][0]+"?start="+urls[ele][1]+"&end="+urls[ele][2]
            print("# " + str(ele)+" Downloading: " + actual_url)
            # actually it will download the whole dataset
            ydl.download([actual_url])
            # cut the video
            original_file_name = path+'/{0}/{0}_{1}.wav'.format(label,urls[ele][0])
            modified_file_name = path+'/{0}/{0}_{1}.wav'.format(label+"Clip", urls[ele][0])
            print("Original file: " + original_file_name)
            print("Clip file: " + modified_file_name)
            cmd = "ffmpeg -i " + original_file_name + " -ss " + urls[ele][1] + " -to "+ urls[ele][2] +" -c copy " + modified_file_name
            os.system(cmd)


def download():
    # read ids
    label_json = "../ontology.json"
    labels = json.load(open(label_json))
    label_dict = dict()
    for label in labels:
        label_dict[label["name"]] = label["id"]


    label_names = list([Speech])
    folder_names = list(["Speech"])
    for item in range(len(label_names)):
        label_urls = read_urls(label_dict[label_names[item]], all_files)
        get_audio_data(label_urls, "./data", folder_names[item])


# download()


# split data
def cut_audio_data(input_path, output_path):
    for fn in glob.glob(os.path.join(input_path, "*.wav")):
        file_name = fn.split("/")[-1].split(".")[0]
        print(fn)
        for start_time in range(10):
            cmd = "ffmpeg -i " + fn + " -ss " + str(start_time) + " -to "+ str(start_time+1) + " -c copy " + os.path.join(output_path, "OneSecond"+file_name + str(start_time) + ".wav")
            print(cmd)
            os.system(cmd)


cut_audio_data("./data/CoughClip/", "./OneSecondData/CoughClip/")



