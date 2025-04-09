import numpy as np
from scipy.signal import convolve2d
from skimage.color import rgb2gray

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#


def get_meshgrid(sigma):
    kernel_size = np.round(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1
    half_size = np.floor(kernel_size / 2)
    return np.meshgrid(np.arange(-half_size, half_size + 1), np.arange(-half_size, half_size + 1))


def laplacian_of_gaussian_filter(sigma):
    x, y = get_meshgrid(sigma)
    sigma_squared = sigma ** 2
    sum_squared = x ** 2 + y ** 2
    term = -(1 / sigma_squared) * (1 - (sum_squared / (2 * sigma_squared)))
    exp_term = np.exp(-sum_squared / (2 * sigma_squared))
    kernel = term * exp_term
    kernel = kernel / np.sum(np.abs(kernel))
    return kernel


def difference_of_gaussian_filter(sigma, k):
    x, y = get_meshgrid(sigma)
    sigma_squared = sigma ** 2
    sum_squared = x ** 2 + y ** 2

    gaussian_1 = np.exp(-sum_squared / (2 * sigma_squared * (k**2)))
    gaussian_1 = gaussian_1 / np.sum(np.abs(gaussian_1))

    gaussian_2 = np.exp(-sum_squared / (2 * sigma_squared))
    gaussian_2 = gaussian_2 / np.sum(np.abs(gaussian_2))

    return gaussian_1 - gaussian_2


def create_scale_space(image, method, k, initial_sigma, levels):
    h, w = np.shape(image)
    scale_space = np.zeros((h, w, levels), np.float32)
    sigma = [0] * (levels + 1)
    sigma[0] = initial_sigma
    for i in range(levels):
        # print(f"{i} Getting kernel for sigma={sigma[i]}")
        if method == 'LOG':
            kernel = laplacian_of_gaussian_filter(sigma[i])
        else:
            kernel = difference_of_gaussian_filter(sigma[i], k)
        # print(f"{i} Completed kernel for sigma={sigma[i]}")
        convolved_image = convolve2d(image, kernel, mode='same')
        scale_space[:, :, i] = np.square(convolved_image)
        sigma[i + 1] = sigma[i] * k
        print(f"\rCompleted {(i + 1) * 100 / levels:.2f} %", end=" ", flush=True)
    print("\r", flush=True)
    return scale_space, sigma


def max_suppression(scale_space, sigma, threshold, levels):
    radius = [0] * levels
    index = [(1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for i in range(levels):
        radius[i] = int(np.ceil(np.sqrt(2) * sigma[i]))

    size = np.shape(scale_space[:, :, 0])

    def check(l):
        return all(scale_space[i + dx, j + dy, l] < scale_space[i, j, k]
                   for dx, dy in index
                   if 0 <= i + dx < size[0] and 0 <= j + dy < size[1])

    blob_location = []
    for k in range(levels):
        for i in range(radius[k] + 1, size[0] - radius[k] - 1):
            for j in range(radius[k] + 1, size[1] - radius[k] - 1):
                if scale_space[i, j, k] < threshold:
                    continue
                c_max = check(k)
                l_max = u_max = True
                if k - 1 >= 0:
                    l_max = check(k - 1) and scale_space[i, j, k - 1] < scale_space[i, j, k]
                if k + 1 < levels:
                    u_max = check(k + 1) and scale_space[i, j, k + 1] < scale_space[i, j, k]

                if c_max and l_max and u_max:
                    blob_location.append((j, i, radius[k]))
    return blob_location


def edge_detector(locations, scale_space, sigma, level):
    sobelFilter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    # Gaussian filtering for smoothing
    blob_locations = []
    for k in range(level):

        image_grad_x = convolve2d(scale_space[:, :, k], sobelFilter, mode='same')
        image_grad_y = convolve2d(scale_space[:, :, k], sobelFilter.T, mode='same')
        kernel = laplacian_of_gaussian_filter(sigma[k])
        image_x = convolve2d(np.square(image_grad_x), kernel, mode='same')
        image_y = convolve2d(np.square(image_grad_y), kernel, mode='same')
        image_xy = convolve2d(image_grad_x * image_grad_y, kernel, mode='same')

        det = (image_x * image_y) - (image_xy ** 2)
        trace = image_x + image_y
        R = det - 0.05 * (trace ** 2)
        for loc in locations:
            if loc[2] == k and R[loc[1], loc[0]] > 0:
                blob_locations.append(loc)

    # print(f"Removed {len(locations) - len(blob_locations)} blobs from harris edge detector")
    return blob_locations


def detectBlobs(im, params):
    # Input:
    #   IM - input image
    #
    # Ouput:
    #   BLOBS - n x 5 array with blob in each row in (x, y, radius, angle, score)
    #
    # Dummy - returns a blob at the center of the image

    im = rgb2gray(im)
    scale_space, sigma = create_scale_space(im, params.filter, params.k, params.initial_sigma, params.levels)
    locations = max_suppression(scale_space, sigma, params.threshold, params.levels)

    # locations = edge_detector(locations, scale_space, sigma, params.levels)

    blobs = []
    for loc in locations:
        blobs.append([loc[0], loc[1], loc[2], 100, 1.0])

    return np.array(blobs)
