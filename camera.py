import picamera
import picamera.array
import pygame
import pygame.camera
from pygame.locals import *
from time import sleep


class Camera:
    def save_image(self, path):
        raise NotImplementedError

    def get_img_array(self):
        raise NotImplementedError


class PiCamera(Camera):
    def __init__(self, img_size=(150, 150)):
        print('Starting RPI camera...')
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


class UsbCamera:
    def __init__(self, img_size=(150, 150)):
        print('Starting USB camera...')
        pygame.init()
        pygame.camera.init()
        self.camera = pygame.camera.Camera("/dev/video0", img_size)
        self.camera.start()
        sleep(2)
        print('Camera is ready to be used')

    def save_image(self, path):
        print(f'Capturing image to {path}')
        image = self.camera.get_image()
        pygame.image.save(image, path)

    def get_img_array(self):
        img = self.camera.get_image()
        return pygame.surfarray.array3d(img)

    def __del__(self):
        self.camera.stop()
        print('Camera has stopped')

if __name__ == '__main__':
    c = UsbCamera((500, 500))
    # c = PiCamera((500, 500))
    img = c.get_img_array()
    c.save_image('tmp.jpg')
    print(img.shape)
