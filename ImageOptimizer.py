import matplotlib.pyplot as plt
import numpy as np
import imageio

target_image = imageio.imread(r"C:\Users\thehu\Documents\ImageOptimizer\Untitled.png")
current_image = 255*np.ones(shape=target_image.shape, dtype=np.uint8)

def measure_error(target_image, image):
    return np.sum(np.abs(target_image-image).flatten())

def add_random_shape(image):

    new_image = image.copy()

    shape = np.random.choice(['Square', 'Circle'])
    [im_x_size, im_y_size, _] = image.shape

    if shape == 'Square':
        x0 = np.random.randint(0, im_x_size)
        y0 = np.random.randint(0, im_y_size)

        x_size = np.random.randint(0, 1+min(x0, max(im_x_size - x0, 0)))
        y_size = np.random.randint(0, 1+min(y0, max(im_y_size - y0, 0)))

        x_min, x_max = max(x0 - x_size, 0), min(im_x_size - 1, x0 + x_size)
        y_min, y_max = max(y0 - y_size, 0), min(im_y_size - 1, y0 + y_size)

        color = np.random.randint(0, 255, size=3)

        new_image[x_min:x_max, y_min:y_max, :] = color[np.newaxis, np.newaxis, :]
    elif shape == 'Circle':
        center_x = np.random.randint(1, im_x_size-1)
        center_y = np.random.randint(1, im_y_size-1)

        radius = np.random.randint(0, min(center_x, center_y, im_x_size - center_x, im_y_size - center_y))

        color = np.random.randint(0, 255, size=3)

        for x in range(center_x-radius, center_x+radius+1):
            for y in range(center_y-radius, center_y+radius+1):
                if (center_x - x) ** 2 + (center_y - y) ** 2 < radius ** 2:
                    new_image[x, y] = color[np.newaxis, np.newaxis, :]

    new_error = measure_error(target_image, new_image)

    return new_image, new_error

N = 100000000
for i in range(N):
    old_error = measure_error(target_image, current_image)

    best_error = old_error
    candidate_image = current_image

    for x in range(5):

        new_image, error = add_random_shape(current_image)

        if error < best_error:
            best_error = error
            candidate_image = new_image

    current_image = candidate_image

    print("{}/{}".format(i, N))

    if i % 100 == 0:
        plt.imshow(current_image)
        plt.imsave("IM_{}.png".format(i), arr=current_image)