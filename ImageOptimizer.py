from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import imageio
import os

from pylab import *
from mpl_toolkits.mplot3d import Axes3D

class ImageOptimizer:

    def __init__(self, target_image_path, seed_image_path=None, project_name=None):

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
        if project_name is None:
            import time
            self.project_name = "Project - {}/".format(int(time.time()))
        else:
            if project_name[-1] == r"\\" or project_name[-1] == r"/":
                self.project_name = project_name
            else:
                self.project_name = project_name + r"/"
        self.backup_folder = "generated_images/"
        if not os.path.exists(self.backup_folder+self.project_name):
            os.makedirs(self.backup_folder+self.project_name)

        #Supported operations
        self.supported_operations = ['Square', 'Circle', 'GaussBlur']

        #Running average success rate estimation for the supported operations
        self.running_average_length = 200
        self.successes = OrderedDict()
        self.success_rates = OrderedDict()
        for operation in self.supported_operations:
            self.successes[operation] = [True] * self.running_average_length
            self.success_rates[operation] = 1 / len(self.supported_operations)

        #Keeping track of which colors are the most successful
        self.n_color_bins = 32
        assert 256 % self.n_color_bins == 0, "256 must be evenly divisible by number of color bins"
        self.color_scores_binned = 10 * np.ones(shape=(self.n_color_bins, self.n_color_bins, self.n_color_bins))
        self.color_scores_normalization_rate = 0.99
        self.color_score_blur_window_size = 7
        self.color_increase_rate = 10

        #Keeping track of which positions are the most successful
        self.position_scores = 10 * np.ones(shape=self.target_image.shape[0:2])
        self.position_scores_normalization_rate = 0.99
        self.position_scores_sigma = 50
        self.position_increase_rate = 30

    def make_image_rgb(self, image):

        if len(image.shape) == 2:   #If black and white image convert to RGB
            return np.stack([image, image, image], axis=2)
        elif image.shape[2] > 3:    #If alpha channel, remove it
            return image[:, :, :3]
        else:
            return image

    def measure_error(self, image):
        return np.sum(np.abs(self.target_image-image))

    def generate_color_sampling_probs_binned(self):
        return np.exp(self.color_scores_binned) / np.sum(np.exp(self.color_scores_binned))

    def generate_marginal_color_probs_binned(self):
        color_probs_binned = self.generate_color_sampling_probs_binned()
        r_probs = np.squeeze(np.sum( np.sum(color_probs_binned, axis=1, keepdims=True), axis=2, keepdims=True))
        g_probs = np.squeeze(np.sum( np.sum(color_probs_binned, axis=0, keepdims=True), axis=2, keepdims=True))
        b_probs = np.squeeze(np.sum( np.sum(color_probs_binned, axis=0, keepdims=True), axis=1, keepdims=True))
        return r_probs, g_probs, b_probs

    def generate_center_sampling_probs(self):
        return np.exp(self.position_scores) / np.sum(np.exp(self.position_scores))

    def update_color_scores(self, color, mean_improvement):
        [r, g, b] = color[..., :]

        def find_correct_color_bin(r, g, b, n_bins):
            return int(r/256*n_bins), int(g/256*n_bins), int(b/256*n_bins)

        ri, gi, bi = find_correct_color_bin(r, g, b, self.n_color_bins)

        # Updating color scores
        self.color_scores_binned[ri, gi, bi] += self.color_increase_rate * (1 + mean_improvement)

        # Normalizing error with a moving box filter
        window_size_half = int(self.color_score_blur_window_size/2)
        new_scores = self.color_scores_binned.copy()
        for ri in range(0, self.n_color_bins):
            for gi in range(0, self.n_color_bins):
                for bi in range(0, self.n_color_bins):

                    r_min, r_max = max(0, ri - window_size_half), min(self.n_color_bins, ri + window_size_half)
                    g_min, g_max = max(0, gi - window_size_half), min(self.n_color_bins, gi + window_size_half)
                    b_min, b_max = max(0, bi - window_size_half), min(self.n_color_bins, bi + window_size_half)

                    new_scores[ri, gi, bi] = np.mean(
                        self.color_scores_binned[r_min:r_max,
                                                 g_min:g_max,
                                                 b_min:b_max])

        self.color_scores_binned = new_scores.copy()
        self.color_scores_binned *= self.color_scores_normalization_rate

    def update_position_scores(self, center, mean_improvement):
        [x, y] = center

        # Updating position scores
        self.position_scores[x, y] += self.position_increase_rate * (1 + mean_improvement)

        # Normalizing error with a gaussian blur and proportional reductions
        self.position_scores = gaussian_filter(self.position_scores, sigma=self.position_scores_sigma)
        self.position_scores *= self.position_scores_normalization_rate

    def sample_operation(self):
        success_rates_norm = np.array([success_rate for success_rate in self.success_rates.values()])
        success_rates_norm = np.exp(success_rates_norm) / np.sum(np.exp(success_rates_norm))
        return np.random.choice(self.supported_operations, p=success_rates_norm)

    def sample_color(self):

        color_probs = self.generate_color_sampling_probs_binned()

        r_probs = np.sum(np.sum(color_probs, axis=2, keepdims=False), axis=1, keepdims=False)
        r = np.random.choice(range(0, self.n_color_bins), p=r_probs)

        g_probs = np.sum(color_probs[r, :, :], axis=1, keepdims=False)
        g = np.random.choice(range(0, self.n_color_bins), p=g_probs/np.sum(g_probs))

        b_probs = color_probs[r, g, :]
        b = np.random.choice(range(0, self.n_color_bins), p=b_probs/np.sum(b_probs))

        n_val_per_bin = 256 / self.n_color_bins
        r = r * n_val_per_bin + np.random.randint(0, n_val_per_bin)
        g = g * n_val_per_bin + np.random.randint(0, n_val_per_bin)
        b = b * n_val_per_bin + np.random.randint(0, n_val_per_bin)

        return np.array([int(r), int(g), int(b)])

    def sample_position(self):

        pos_probs = self.generate_center_sampling_probs()

        [im_x_size, im_y_size, _] = self.target_image.shape

        x = np.random.choice(range(0, im_x_size), p=np.sum(pos_probs, axis=1, keepdims=False))
        y = np.random.choice(range(0, im_y_size), p=pos_probs[x, :]/np.sum(pos_probs[x, :]))

        return x, y


    def perform_random_operation(self, image):

        new_image = image.copy()

        [im_x_size, im_y_size, _] = image.shape
        operation = self.sample_operation()
        color =  None
        x0, y0 = None, None

        if operation == 'Square':

            #Sampling center location
            x0, y0 = self.sample_position()

            #Sampling size
            x_size = np.random.randint(0, 1+min(x0, max(im_x_size - x0, 0)))
            y_size = np.random.randint(0, 1+min(y0, max(im_y_size - y0, 0)))

            #Assuring that the shape fits the image
            x_min, x_max = max(x0 - x_size, 0), min(im_x_size - 1, x0 + x_size)
            y_min, y_max = max(y0 - y_size, 0), min(im_y_size - 1, y0 + y_size)

            #Sampling color
            color = self.sample_color()

            #Creating new image
            new_image[x_min:x_max, y_min:y_max, :] = color[np.newaxis, np.newaxis, :]

        elif operation == 'Circle':

            #Sampling center location
            x0, y0 = self.sample_position()

            #Circles should not be along the edge of images
            if x0 == im_x_size:
                x0 -= 1
            elif x0 == 0:
                x0 += 1

            if y0 == im_y_size:
                y0 -= 1
            elif y0 == 0:
                y0 += 1

            #Sampling radius and color
            radius = np.random.randint(0, min(x0, y0, im_x_size - x0, im_y_size - y0))
            color = self.sample_color()

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

        return new_image, operation, color, (x0, y0)

    def estimate_target_image(self, N=10000000000, backup_freq=100, web_image_freq=10):

        #Estimating target image
        web_updated, backup_updated = False, False
        for i in range(N):

            #Logging current error
            current_error = self.measure_error(self.current_image)

            #Performing random operation and calculating new error
            new_image, operation, color, center = self.perform_random_operation(self.current_image)
            new_error = self.measure_error(new_image)
            success = new_error < current_error

            #Replace current image if the error is reduced
            if success:

                #The backup and web images should be updated
                web_updated, backup_updated = True, True

                #Replace image
                self.current_image = new_image

                #Count success
                self.successes[operation].append(True)
                self.successes[operation].pop(0)

                # How big was the success? Use the mean pixel improvement as a measure for this
                [im_x_size, im_y_size, n_channels] = self.target_image.shape
                mean_improvement = abs(current_error - new_error) / (im_x_size * im_y_size * n_channels)

                #If a color was used keep track of it
                if color is not None:
                    self.update_color_scores(color, mean_improvement)

                #If a center position was used keep track of it
                if center[0] is not None and center[1] is not None:
                    self.update_position_scores(center, mean_improvement)

            else:
                self.successes[operation].append(False)
                self.successes[operation].pop(0)

            #Estimating success rates
            success_count = sum([int(x) for x in self.successes[operation]])
            self.success_rates[operation] = success_count / self.running_average_length

            #Printing information
            print("{}/{}: {:<12}, {:<6}, {:<4} : {:<12}".format(i, N, operation[:19], str(success),
                                                                round(self.success_rates[operation], 2), int(new_error)))

            #Updating web file
            if i % web_image_freq == 0 and web_updated:
                plt.imsave(self.backup_folder+"web_image.png", arr=self.current_image.astype('uint8'))

                plt.clf()
                r_probs, g_probs, b_probs = self.generate_marginal_color_probs_binned()
                plt.plot(np.arange(0, 32), r_probs, color='r')
                plt.plot(np.arange(0, 32), g_probs, color='g')
                plt.plot(np.arange(0, 32), b_probs, color='b')
                plt.grid()
                plt.title("RGB sampling preferences")
                plt.savefig(self.backup_folder+"web_rgb.png")

                plt.imsave(self.backup_folder+"web_center.png", arr=self.generate_center_sampling_probs())

                web_updated = False

            #Updating backup
            if i % backup_freq == 0 and backup_updated:
                plt.imsave(self.backup_folder+self.project_name+"IM_{}.png".format(i),
                           arr=self.current_image.astype('uint8'))


                plt.clf()
                r_probs, g_probs, b_probs = self.generate_marginal_color_probs_binned()
                plt.plot(np.arange(0, 32), r_probs, color='r')
                plt.plot(np.arange(0, 32), g_probs, color='g')
                plt.plot(np.arange(0, 32), b_probs, color='b')
                plt.grid()
                plt.savefig(self.backup_folder+self.project_name+"RGB_{}.png".format(i))

                plt.imsave(self.backup_folder + self.project_name + "CENTER_{}.png".format(i), arr=self.generate_center_sampling_probs())

                backup_updated = False



            #3D RGB VIS
            def vis(n_bins, color_probs):

                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')

                N = 4000
                rs = np.random.randint(0, n_bins, N)
                gs = np.random.randint(0, n_bins, N)
                bs = np.random.randint(0, n_bins, N)
                p = color_probs[rs, gs, bs]
                p_norm = (p - np.min(p)) / np.max((p - np.min(p)))

                colors = cm.viridis(p_norm)
                colmap = cm.ScalarMappable(cmap=cm.viridis)
                colmap.set_array(p_norm)

                ax.scatter(xs=rs, ys=gs, zs=bs, c=colors, marker='o')
                cb = fig.colorbar(colmap)

                ax.set_xlabel('RED')
                ax.set_ylabel('GREEN')
                ax.set_zlabel('BLUE')

                plt.show()

            if i == 20000:
                vis(self.n_color_bins, self.generate_color_sampling_probs_binned())