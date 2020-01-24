from camera import PiCamera
from camera import UsbCamera
from datetime import datetime
import numpy as np
from PIL import Image
from train_model import ModelTrainer
from datetime import datetime
import os
import time


class ImagePredictor:
    def __init__(self, training_path, predictions_dir=None):
        # self.camera = UsbCamera(img_size=(50, 50))
        self.camera = PiCamera(img_size=(50, 50))
        trainer = ModelTrainer(training_path)
        if not os.path.exists('model.h5'):
                trainer.train(epochs=5)
                trainer.save_weights('model.h5')
        trainer.load_weights('model.h5')
        self.model = trainer.get_model()
        self.label_dict = trainer.get_label_dict()
        self.folder = predictions_dir
        if self.folder:
            if not os.path.isdir(self.folder):
                os.mkdir(self.folder)

    def predict_image(self, save=False):
        img = self.camera.get_img_array()
        prediction = self.model.predict(np.array([img]))[0]
        # print(predictions)
        label = np.argmax(prediction)
        # print(label)
        if save:
            if not self.folder:
                raise ValueError('Please set folder name where to save the pictures!')
            self.__save_img(img, self.label_dict[label])
        return self.label_dict[label]

    def __save_img(self, img_array, label):
        now = datetime.now()
        dt_str = now.strftime(f'%Y-%m-%d_%H-%M-%S')
        file_name = f'{dt_str}_{label}.png'
        print(f'Save file: {file_name} to folder {self.folder}')
        img = Image.fromarray(img_array)
        img.save(os.path.join(self.folder, file_name))


if __name__ == '__main__':
    predictor = ImagePredictor('training', predictions_dir='predictions')
    for _ in range(10):
        start = time.perf_counter()
        prediction = predictor.predict_image(save=True)
        running_time = time.perf_counter() - start
        print(f'{prediction} (exec time: {"{0:.3f}".format(running_time)} sec)')
        time.sleep(2)
