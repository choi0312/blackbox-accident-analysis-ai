import random
import numpy as np

from PIL import Image, ImageFilter, ImageEnhance


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = [max(1 - brightness, 0), 1 + brightness]
        self.contrast = [max(1 - contrast, 0), 1 + contrast]
        self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, img):

        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)
        return img


class RandomGaussianBlur(object):
    def __init__(self, radius=1):
        self.radius = radius

    def __call__(self, img):
        if random.random() < 0.5:
            img = img.filter(
                ImageFilter.GaussianBlur(radius=self.radius * random.random())
            )

        return img


class RandomGaussianNoise(object):
    def __init__(self, mean=0, sigma=10):
        self.mean = mean
        self.sigma = sigma

    def gaussianNoisy(self, im, mean=0, sigma=10):
        # Generate noise based on the input image shape
        noise = np.random.normal(mean, sigma, im.shape)
        im = im + noise
        im = np.clip(im, 0, 255)
        return im

    def __call__(self, img):
        if random.random() < 0.5:  # Only apply noise with 50% probability
            img = np.asarray(img)

            # Detect if the image is grayscale or RGB based on the number of channels
            if img.ndim == 2:  # Grayscale image
                img = self.gaussianNoisy(img, self.mean, self.sigma)
            elif img.ndim == 3:  # RGB image
                img = img.astype(float)  # Convert to float for processing
                for c in range(3):  # Apply noise to each channel
                    img[:, :, c] = self.gaussianNoisy(
                        img[:, :, c], self.mean, self.sigma
                    )

            img = Image.fromarray(img.astype(np.uint8))

        return img
