import picamera
import picamera.array
from time import sleep
# from keras.preprocessing.image import save_img


class Camera:
    def __init__(self, img_size=(150, 150)):
        print('Starting camera...')
        self.camera = picamera.PiCamera()
        self.camera.resolution = img_size
        self.camera.start_preview()
        sleep(2)
        print('Camera is ready to be used')

    def save_image(self, path):
        print(f'Capturing image to {path}')
        self.camera.capture(path)

    def get_img_array(self):
        with picamera.array.PiRGBArray(self.camera) as stream:
            self.camera.capture(stream, 'rgb')
            return stream.array

    def __del__(self):
        self.camera.stop_preview()
        print('Camera has stopped')


if __name__ == '__main__':
    c = Camera()
    # c.save_image('tmp.jpg')
    img = c.get_img_array()
    print(img.shape)
    # save_img('tmp_keras.jpg', img)
