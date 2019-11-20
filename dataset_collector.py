import argparse
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
import numpy as np
import os
import tensorflow as tf

class DatasetCollector:
    def __init__(self):
        self.label_dict = dict()

    def get_data(self, data_folder, target_size):
        image_list = []
        label_list = []

        i = 0
        label = 0
        for folder_name in os.listdir(data_folder):
            for filename in os.listdir(os.path.join(data_folder, folder_name)):
                img = load_img(os.path.join(data_folder, folder_name, filename),
                               target_size=target_size)
                img_array = img_to_array(img)
                image_list.append(img_array)
                label_list.append(label)
                i += 1
                if i % 1000 == 0:
                    print(i)
            self.label_dict[label] = folder_name
            label += 1

        images = tf.Session().run(tf.random_shuffle(np.array(image_list), seed=8))
        labels = tf.Session().run(tf.random_shuffle(np.array(label_list), seed=8))
        return images/255, to_categorical(labels)


def create_argparser():
    parser = argparse.ArgumentParser(description='Dataset collector')
    parser.add_argument('-d', '--data_dir', required=True, help='Dataset folder')
    parser.add_argument('-s', '--size', default='150,150', help='Target image size, for example "150,150"')
    return parser


if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()
    size = tuple([int(i) for i in args.size.split(',')])

    collector = DatasetCollector
    data, labels = collector.get_data(args.data_dir, size)
