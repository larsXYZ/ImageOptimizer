from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import imageio
import multiprocessing as mp

target_image = imageio.imread(r"current_target.png")
current_image = 255*np.ones(shape=target_image.shape, dtype=np.uint8)

def measure_error(target_image, image):
    return np.sum(np.abs(target_image-image).flatten())

def perform_random_operation(image):

    new_image = image.copy()

    operations = [['Square', 0.45], ['Circle', 0.45], ['GaussBlur', 0.1]]
    rand = np.random.random()
    
    for x in operations:
        if rand < x[1]:
            operation = x[0]
            break
        rand -= x[1]
    
    [im_x_size, im_y_size, _] = image.shape

    if operation == 'Square':
        x0 = np.random.randint(0, im_x_size)
        y0 = np.random.randint(0, im_y_size)

        x_size = np.random.randint(0, 1+min(x0, max(im_x_size - x0, 0)))
        y_size = np.random.randint(0, 1+min(y0, max(im_y_size - y0, 0)))

        x_min, x_max = max(x0 - x_size, 0), min(im_x_size - 1, x0 + x_size)
        y_min, y_max = max(y0 - y_size, 0), min(im_y_size - 1, y0 + y_size)

        color = np.random.randint(0, 255, size=3)

        new_image[x_min:x_max, y_min:y_max, :] = color[np.newaxis, np.newaxis, :]
    elif operation == 'Circle':
        center_x = np.random.randint(1, im_x_size-1)
        center_y = np.random.randint(1, im_y_size-1)

        radius = np.random.randint(0, min(center_x, center_y, im_x_size - center_x, im_y_size - center_y))

        color = np.random.randint(0, 255, size=3)

        for x in range(center_x-radius, center_x+radius+1):
            for y in range(center_y-radius, center_y+radius+1):
                if (center_x - x) ** 2 + (center_y - y) ** 2 < radius ** 2:
                    new_image[x, y] = color[np.newaxis, np.newaxis, :]
    elif operation == 'GaussBlur':
        sigma = np.random.randint(0, 10)
        new_image = gaussian_filter(new_image, sigma=sigma)

    return new_image, operation
    
def perform_random_operations(current_image, target_image):

    for i in range(1000):
    
        current_error = measure_error(target_image, current_image)
        new_image, operation = perform_random_operation(current_image)
        new_error = measure_error(target_image, new_image)
        
        if new_error < current_error:

            current_image = new_image
     
    current_error = measure_error(target_image, current_image)
     
    return current_image, current_error

if __name__ == "__main__":
    
    N = 100000000
    for i in range(N):

        parallel_threads = 20

        pool = mp.Pool(parallel_threads)
            
        results = [pool.apply_async(perform_random_operations, args=(current_image, target_image)) for _ in range(parallel_threads)]
        candidates = [result.get() for result in results]

        current_error = measure_error(target_image, current_image)
        
        for x in candidates:

            if x[1] < current_error:

                current_image = x[0]
                current_error = x[1]

        print("{}/{}".format(i, N))

        if i % 1 == 0:
            plt.imsave("web_image.png", arr=current_image)