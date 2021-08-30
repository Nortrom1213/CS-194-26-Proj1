import numpy as np
import skimage.io as skio
import skimage.filters as skfil
import skimage.transform as sktr
import math

# Sum of Squared Differences
def L2(A, B):
    return np.mean((A-B)**2)

# Normalized Cross Correlation
def NCC(A, B):
    return np.sum((A-np.mean(A))*(B-np.mean(B))) / \
                                (np.sum(np.sqrt((np.sum(A-np.mean(A))**2)*(np.sum(B-np.mean(B))**2)))+1)

# Compute the area given four borders
def area(x):
    return (x[1] - x[0]) * (x[3] - x[2])

# Move the Image by x axis or y axis
def move_by_x(A, n):
    return np.roll(A, n, 1)

def move_by_y(A, n):
    return np.roll(A, n, 0)

def row(x, y):
    return x[y]

def column(x, y):
    return x[:, y]

# Crops an image given the limits
def crop(image, border):
    return image[border[0]:border[1], border[2]:border[3]]

# Find the borders for the image for a specific channel to crop out the unnatural ones
def auto_crop(image):
    img = skfil.sobel(image)
    height = img.shape[0]
    width = img.shape[1]

    # Find the borders
    top = find_border(img, row, lambda x: x, 0)
    bottom = find_border(img, row, lambda x: height-1-x, height)
    left = find_border(img, column, lambda x: x, 0)
    right = find_border(img, column, lambda x: width-1-x, width)

    return top, bottom, left, right

# Select the border that crops out the most
def crop_all(B, G, R):
    auto_crops = [auto_crop(B), auto_crop(G), auto_crop(R)]
    min_crop = min(auto_crops, key=area)

    B = crop(B, min_crop)
    G = crop(G, min_crop)
    R = crop(R, min_crop)

    return B, G, R

def find_border(image, line, position, default):
    max_move = int(image.shape[1]/20)

    border = default
    for i in range(0, max_move):
        val = np.sum(line(image, position(max_move - i)))
        # Choose the First val larger than value threshold as the border
        if val >= len(image)/100:
            border = position(max_move - i)
            break

    return border

# Exhaust Search in [-20. 20] range with NCC or L2 as dist func
# Return the result with the best val
def exhaust_align(A, B):
    moves = []
    for i in range(-20, 20):
        for j in range(-20, 20):
            newA = move_by_y(move_by_x(A, i), j)
            val = method(newA, B)
            moves.append([val, [i, j]])

    sorted_moves = sorted(moves, key=lambda x: x[0], reverse= method == NCC)
    return sorted_moves[0][1]

# Exhaust search
def exhaust_search(original_image, save_name, should_crop):
    img = skio.imread(original_image)
    height = np.floor(img.shape[0] / 3.0).astype(np.int)

    # Splits the image
    B = img[:height]
    G = img[height:2 * height]
    R = img[2 * height:3 * height]

    if should_crop:
        B, G, R = crop_all(B, G, R)

    # Blue Base : It does not matter to choose which channel as the base channel
    GB = exhaust_align(G, B)
    G = move_by_x(move_by_y(G, GB[1]), GB[0])
    RG = exhaust_align(R, G)
    R = move_by_x(move_by_y(R, RG[1]), RG[0])

    if should_crop:
        B, G, R = crop_all(B, G, R)

    final = np.dstack([R, G, B])
    skio.imsave(save_name, final)

def recurs(R, G, B, exp, min, moves):
    if exp <= min:
        return [R, G, B], moves

    scale = 2**exp
    Red = sktr.rescale(R, 1 / scale)
    Green = sktr.rescale(G, 1 / scale)
    Blue = sktr.rescale(B, 1 / scale)

    GB = exhaust_align(Green, Blue)
    Green = move_by_x(move_by_y(Green, GB[1]), GB[0])
    RG = exhaust_align(Red, Green)

    # Scale Back
    R = move_by_x(move_by_y(R, RG[1] * scale), RG[0] * scale)
    G = move_by_x(move_by_y(G, GB[1] * scale), GB[0] * scale)

    # Record all moves and show them in show_displacement
    moves.append([np.multiply(RG, scale), np.multiply(GB, scale)])

    return recurs(R, G, B, exp - 1, min, moves)

def pyramid_search(original_image, save_name, should_crop):
    img = skio.imread(original_image)
    height = np.floor(img.shape[0] / 3.0).astype(np.int)

    # Splits the image
    B = img[:height]
    G = img[height:2*height]
    R = img[2*height:3*height]

    if should_crop:
        B, G, R = crop_all(B, G, R)

    exponent = int(math.log2(B.shape[1]/100))
    min_exponent = 0 if exponent <= 1 else 1

    moves = []
    RGB, moves = recurs(R, G, B, exponent, min_exponent, moves)
    red_dis = [0, 0]
    green_dis = [0, 0]
    for move in moves:
        red_dis = np.add(red_dis, move[0])
        green_dis = np.add(green_dis, move[1])
    print("red displacement:[", red_dis[0], red_dis[1], "] green displacement:[", green_dis[0], green_dis[1], "]")

    R = RGB[0]
    G = RGB[1]
    B = RGB[2]

    if should_crop:
        B, G, R = crop_all(B, G, R)

    final = np.dstack([R, G, B])
    skio.imsave(save_name, final)

input = './cathedral.jpg'
method = NCC
should_crop = 1
pyramid = 1
if pyramid:
    output = './cathedral_pyramid_NCC.jpg'
    pyramid_search(input, output, should_crop)
else:
    output = './church_naive_NCC.jpg'
    exhaust_search(input, output, should_crop)

