from camera import PiCamera
from camera import UsbCamera
import numpy as np
from train_model import ModelTrainer
from datetime import datetime
import os
import time


class ImagePredictor:
    def __init__(self, training_path):
        self.camera = UsbCamera(img_size=(50, 50))
        trainer = ModelTrainer(training_path)
        if not os.path.exists('model.h5'):
                trainer.train(epochs=5)
                trainer.save_weights('model.h5')
        trainer.load_weights('model.h5')
        self.model = trainer.get_model()
        self.label_dict = trainer.get_label_dict()

    def predict_image(self):
        img = self.camera.get_img_array()
        prediction = self.model.predict(np.array([img]))[0]
        # print(predictions)
        label = np.argmax(prediction)
        # print(label)
        return self.label_dict[label]


if __name__ == '__main__':
    predictor = ImagePredictor('training')
    for _ in range(10):
        start = time.perf_counter()
        prediction = predictor.predict_image()
        running_time = time.perf_counter() - start
        print(f'{prediction} (exec time: {"{0:.3f}".format(running_time)} sec)')
        time.sleep(2)
