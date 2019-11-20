import argparse
import fileutils
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os


def get_images(directory, target_size):
  result = []
  for filename in os.listdir(directory):
    img = load_img(os.path.join(directory, filename), target_size=target_size)
    img_array = img_to_array(img)
    result.append(img_array)
  return np.array(result)

def create_datagen():
    return ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


def generate_data(directory, target_folder, target_size, limit):
    datagen = create_datagen()
    fileutils.clean_folder(target_folder)
    for folder_name in os.listdir(directory):
        print(f'Generate for {folder_name}')
        os.mkdir(os.path.join(target_folder, folder_name))
        images = get_images(os.path.join(directory, folder_name), target_size=target_size)
        i = 0
        for _ in datagen.flow(images, batch_size=1,
                                  save_to_dir=os.path.join(target_folder, folder_name),
                                  save_prefix=folder_name,
                                  save_format='jpg'):
            i += 1
            if i % 1000 == 0:
                print(f'{folder_name}: {i}')
            if i > limit:
                break


def create_argparser():
    parser = argparse.ArgumentParser(description='Traiing set generator')
    parser.add_argument('-d', '--data_dir', required=True, help='Source folder')
    parser.add_argument('-t', '--target_dir', required=True, help='Target dir')
    parser.add_argument('-s', '--size', default='150,150', help='Target image size')
    parser.add_argument('-l', '--limit', type=int, default=1000, help='Number of generated pictures per class')
    return parser


if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()
    size = tuple([int(i) for i in args.size.split(',')])
    generate_data(args.data_dir, args.target_dir, size, args.limit)
