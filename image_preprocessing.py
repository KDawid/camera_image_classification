import argparse
import fileutils
from keras.preprocessing.image import img_to_array, load_img, save_img
import os


def save_small_images(directory, target_folder, target_size, rotate):
    for filename in os.listdir(directory):
        img = load_img(os.path.join(directory, filename), target_size=target_size)
        if rotate:
            img = img.rotate(-90)
        img_array = img_to_array(img)
        save_img(os.path.join(target_folder, filename), img_array)


def compress_data_set(origin_folder, target_folder, target_size=(150, 150), rotate=None):
    fileutils.clean_folder(target_folder)
    for folder_name in os.listdir(origin_folder):
        if os.path.isdir(os.path.join(origin_folder, folder_name)):
            os.mkdir(os.path.join(target_folder, folder_name))
            save_small_images(os.path.join(origin_folder, folder_name),
                              os.path.join(target_folder, folder_name),
                              target_size,
                              rotate)


def create_argparser():
    parser = argparse.ArgumentParser(description='Create data sets')
    parser.add_argument('-d', '--data_dir', required=True, help='Source folder')
    parser.add_argument('-t', '--target_dir', required=True, help='Target dir')
    parser.add_argument('-s', '--size', default='150,150', help='Target image size')
    parser.add_argument('-r', '--rotate', default=0, help='Rotation value')
    return parser


if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()
    size = tuple([int(i) for i in args.size.split(',')])
    compress_data_set(args.data_dir, args.target_dir, size, args.rotate)
