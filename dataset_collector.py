import argparse
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
import numpy as np
import os


class DatasetCollector:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.label_dict = None

    def get_data(self, target_size):
        image_list = []
        label_list = []

        i = 0
        label = 0
        for folder_name in os.listdir(self.data_folder):
            for filename in os.listdir(os.path.join(self.data_folder, folder_name)):
                img = load_img(os.path.join(self.data_folder, folder_name, filename),
                               target_size=target_size)
                img_array = img_to_array(img)
                image_list.append(img_array)
                label_list.append(label)
                i += 1
                if i % 1000 == 0:
                    print(i)
            label += 1

        images = np.array(image_list)
        labels = np.array(label_list)
        np.random.seed(8)
        np.random.shuffle(images)
        np.random.shuffle(labels)
        return images/255, to_categorical(labels)

    def get_labels_num(self):
        return len(os.listdir(self.data_folder))

    def get_label_dict(self):
        if not self.label_dict:
            self.label_dict = dict()
            label = 0
            for folder_name in os.listdir(self.data_folder):
                self.label_dict[label] = folder_name
                label += 1
        return self.label_dict

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
