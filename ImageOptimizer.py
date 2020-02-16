from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import imageio


class ImageOptimizer:

    def __init__(self, target_image_path, seed_image_path=None):

        #Loading images
        self.target_image = imageio.imread(target_image_path).astype('int')
        if seed_image_path is None:
            self.current_image = 255.0*np.ones(shape=self.target_image.shape, dtype='int')
        else:
            self.current_image = imageio.imread(target_image_path).astype('int')
        self.target_image = self.make_image_rgb(self.target_image)
        self.current_image = self.make_image_rgb(self.current_image)
        assert self.current_image.shape == self.target_image.shape, "Images must have the same dimensions"

        #Folder setup
        self.backup_folder = "generated_images/"

        #Supported operations
        self.supported_operations = ['Square', 'Circle', 'GaussBlur']

        #Running average success rate estimation
        self.running_average_length = 200
        self.successes = OrderedDict()
        self.success_rates = OrderedDict()
        for operation in self.supported_operations:
            self.successes[operation] = [True] * self.running_average_length
            self.success_rates[operation] = 1 / len(self.supported_operations)

    def make_image_rgb(self, image):

        if len(image.shape) == 2:   #If black and white image convert to RGB
            return np.stack([image, image, image], axis=2)
        elif image.shape[2] > 3:    #If alpha channel, remove it
            return image[:, :, :3]
        else:
            return image

    def measure_error(self, image):
        return np.sum(np.abs(self.target_image-image))

    def perform_random_operation(self, image):

        new_image = image.copy()
        success_rates_norm = np.array([ success_rate + 0.01 for success_rate in self.success_rates.values()])
        success_rates_norm = success_rates_norm / np.sum(success_rates_norm)
        operation = np.random.choice(self.supported_operations, p=success_rates_norm )
        [im_x_size, im_y_size, _] = image.shape

        if operation == 'Square':

            #Sampling center location
            x0 = np.random.randint(0, im_x_size)
            y0 = np.random.randint(0, im_y_size)

            #Sampling size
            x_size = np.random.randint(0, 1+min(x0, max(im_x_size - x0, 0)))
            y_size = np.random.randint(0, 1+min(y0, max(im_y_size - y0, 0)))

            #Assuring that the shape fits the image
            x_min, x_max = max(x0 - x_size, 0), min(im_x_size - 1, x0 + x_size)
            y_min, y_max = max(y0 - y_size, 0), min(im_y_size - 1, y0 + y_size)

            #Sampling color
            color = np.random.randint(0, 255, size=3)

            #Creating new image
            new_image[x_min:x_max, y_min:y_max, :] = color[np.newaxis, np.newaxis, :]

        elif operation == 'Circle':

            #Sampling center location
            x0 = np.random.randint(1, im_x_size-1)
            y0 = np.random.randint(1, im_y_size-1)

            #Sampling radius and color
            radius = np.random.randint(0, min(x0, y0, im_x_size - x0, im_y_size - y0))
            color = np.random.randint(0, 255, size=3)

            #Creating new image
            for x in range(x0-radius, x0+radius+1):
                for y in range(y0-radius, y0+radius+1):
                    if (x0 - x) ** 2 + (y0 - y) ** 2 < radius ** 2:
                        new_image[x, y] = color[np.newaxis, np.newaxis, :]

        elif operation == 'GaussBlur':

            #Sampling blur strength
            sigma = np.random.randint(0, 10)

            #Creating new image
            new_image[:, :, 0] = gaussian_filter(new_image[:, :, 0], sigma=sigma)
            new_image[:, :, 1] = gaussian_filter(new_image[:, :, 1], sigma=sigma)
            new_image[:, :, 2] = gaussian_filter(new_image[:, :, 2], sigma=sigma)

        return new_image, operation

    def estimate_target_image(self, N=10000000000, backup_freq=100, web_image_freq=10):

        #Estimating target image
        web_updated, backup_updated = False, False
        for i in range(N):

            #Logging current error
            current_error = self.measure_error(self.current_image)

            #Performing random operation and calculating new error
            new_image, operation = self.perform_random_operation(self.current_image)
            new_error = self.measure_error(new_image)

            #Replace current image if the error is reduced
            if new_error < current_error:
                web_updated, backup_updated = True, True
                self.current_image = new_image
                self.successes[operation].append(True)
                self.successes[operation].pop(0)
            else:
                self.successes[operation].append(False)
                self.successes[operation].pop(0)

            #Estimating success rates
            success_count = sum([int(x) for x in self.successes[operation]])
            self.success_rates[operation] = success_count / self.running_average_length

            #Printing information
            print("{}/{}: {:<12}, {:<6}, {:<4} : {:<12}".format(i, N, operation[:19], str(new_error < current_error),
                                                                round(self.success_rates[operation], 2), int(new_error)))

            #Updating web file
            if i % web_image_freq == 0 and web_updated:
                plt.imsave(self.backup_folder+"web_image.png", arr=self.current_image.astype('uint8'))
                web_updated = False

            #Updating backup
            if i % backup_freq == 0 and backup_updated:
                plt.imsave(self.backup_folder+"IM_{}.png".format(i), arr=self.current_image.astype('uint8'))
                backup_updated = False