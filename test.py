import numpy as np
import skimage.io as skio
import skimage.transform as sktr
import skimage.filters as skfil
import math
import argparse
import time


# Sum of Squared Differences
def SSD(A, B):
    return np.sum(np.sum((A-B)**2))


# Normalized Cross Correlation
def NCC(A, B):
    top = np.sum((A - np.mean(A))*(B - np.mean(B)))
    bottom = np.sum(np.sqrt((np.sum(A - np.mean(A))**2)*(np.sum(B - np.mean(B))**2)))
    return top / (bottom+1)


# Auxiliary function to roll the image horizontally
def move_horizontal(A, n):
    return np.roll(A, n, 1)


# Auxiliary function to roll the image vertically
def move_vertical(A, n):
    return np.roll(A, n, 0)


# Moves the image within a -15 to 15 space and returns
# the movement that results in the best value of func.
# The reverse argument is for NCC, for which we look for the
# maximum number instead of the minimum.
def exhaust_align(A, B, func, reverse):
    list_of_moves = []
    for i in range(-15, 15):
        for j in range(-15, 15):
            moved = move_horizontal(A, i)
            moved = move_vertical(moved, j)
            val = func(moved, B)
            list_of_moves.append([val, [i, j]])

    sorted_list = sorted(list_of_moves, key=lambda x: x[0], reverse=reverse)
    return sorted_list[0][1]


# Sums, formats and prints the accumulate movement
def print_mov_buf(mov_buf):
    red_movement = [0, 0]
    green_movement = [0, 0]
    for mov in mov_buf:
        red_movement = np.add(red_movement, mov[0])
        green_movement = np.add(green_movement, mov[1])
    print("Red movement:\n   x: ", red_movement[0],
          "\n   y: ", red_movement[1],
          "\nGreen movement:\n   x: ", green_movement[0],
          "\n   y: ", green_movement[1])



# Runs a naive search for the optimal movement.
def exhaust_image(original_image, save_name, method, should_crop, should_full_crop):
    image = skio.imread(original_image)
    height = int(len(image)/3)

    # Splits the image
    blue = image[:height]
    green = image[height:2*height]
    red = image[2*height:3*height]

    if should_crop:
        crops = [auto_crop(blue), auto_crop(green), auto_crop(red)]
        min_crop = min(crops, key=lambda x:(x[1]-x[0])*(x[3]-x[2]))

        # Sets new values for the channels
        blue = crop(blue, min_crop)
        green = crop(green, min_crop)
        red = crop(red, min_crop)

    # Sets the reverse value if is going to run NCC
    reversed = method == NCC

    GB = exhaust_align(green, blue, method, reversed)
    # Moves the green channel regarding the blue channel
    green = move_horizontal(move_vertical(green, GB[1]), GB[0])
    RG = exhaust_align(red, green, method, reversed)
    # Moves the red channel regarding the green channel
    red = move_horizontal(move_vertical(red, RG[1]), RG[0])

    print_mov_buf([[RG, GB]])
    final = np.dstack([red, green, blue])
    skio.imsave(save_name, final)


# Pyramid's method recursive function
def recursive_pyramid(R, G, B, mov_buf, method, exponent, min_exponent):
    # If the exponent has reached the minimum,
    # return the final channels and the movement list
    if exponent <= min_exponent:
        return [R, G, B], mov_buf

    # Calculate the downscale factor
    # and rescale the channels
    dif = 2**exponent
    newR = sktr.rescale(R, 1/dif)
    newG = sktr.rescale(G, 1/dif)
    newB = sktr.rescale(B, 1/dif)

    # Sets the reverse value if is going to run NCC
    reversed = method == NCC

    GB = exhaust_align(newG, newB, method, reversed)
    # Moves the downscaled green channel regarding the
    # downscaled blue channel
    newG = move_horizontal(move_vertical(newG, GB[1]), GB[0])
    RG = exhaust_align(newR, newG, method, reversed)

    # Moves the red channel regarding the green channel
    # with the scaling factor as multiplier for the movement
    R = move_horizontal(move_vertical(R, RG[1] * dif), RG[0] * dif)
    # Moves the full scale green channel regarding the blue channel
    # with the scaling factor as multiplier for the movement
    G = move_horizontal(move_vertical(G, GB[1] * dif), GB[0] * dif)

    # Adds scaled movements to ther movement list
    mov_buf.append([np.multiply(RG, dif), np.multiply(GB, dif)])
    # Calls nextjournal iteration with smaller exponent (Bigger image)
    return recursive_pyramid(R, G, B, mov_buf, method, exponent-1, min_exponent)


# Finds an unnatural border and returns its coordinate for
# future trimming
# list_func is the function used to list all the components
# of an axis
# value_funct is the function used to calculate the value of
# a specific axis given an image
def find_border(image, list_func, value_func, default):

    # Value from which we consider the collection of pixels
    # as a border
    value_treshold = len(image)/100
    # Value from which we stop searching for borders
    mov_treshold = int(image.shape[1]/15)

    # Default value set for the found border is the
    # image extreme point
    border = default
    for i in range(0, mov_treshold):
        val = np.sum(list_func(image, value_func(mov_treshold - i)))
        # If the value is GE than the proposed treshold,
        # then we define it as a border and returns it
        if val >= value_treshold:
            border = value_func(mov_treshold - i)
            break
    return border


# Finds all 4 borders for the image to trim the
# unnatural ones
def auto_crop(image):
    # Apply the sobel filter for edge
    # detection
    border_image = skfil.sobel(image)

    # Sets the default values for each extreme
    top = 0
    bottom = border_image.shape[0]
    left = 0
    right = border_image.shape[1]

    # Sets the listing function for the axis
    row = lambda x, y: x[y]
    column = lambda x, y: x[:,y]

    # Finds the borders
    top_row = find_border(border_image, row, lambda x: x, top)
    bottom_row = find_border(border_image, row, lambda x: bottom-1-x, bottom)
    left_col = find_border(border_image, column, lambda x: x, left)
    right_col = find_border(border_image, column, lambda x: right-1-x, right)

    return (top_row, bottom_row, left_col, right_col)


# Evaluates the size of the image
# after being cropped
def crop_eval(x):
    return (x[1]-x[0])*(x[3]-x[2])


# Crops an image given the limits
def crop(img, limits):
    return img[limits[0]:limits[1], limits[2]:limits[3]]


# Starts the recursive process for the pyramid method
def pyramid_search_align(original_image, save_name, method, should_crop, should_full_crop):
    image = skio.imread(original_image)
    height = int(len(image)/3)

    # Splits the image
    blue = image[:height]
    green = image[height:2*height]
    red = image[2*height:3*height]

    if should_crop or should_full_crop:
        crops = [auto_crop(blue), auto_crop(green), auto_crop(red)]
        min_crop = min(crops, key=crop_eval)

        # Sets new values for the channels
        blue = crop(blue, min_crop)
        green = crop(green, min_crop)
        red = crop(red, min_crop)

    # Finds the exponent for which the image will have
    # its X-axis downsized to 100
    exponent = int(math.log2(blue.shape[1]/100))

    # If the image is too small to have more than
    # ones pyramid level, set the minimum exponent
    # to 0
    if exponent <= 1:
        min_exponent = 0
    else:
        min_exponent = 1

    RGB_tuple, mov_buf = recursive_pyramid(red, green, blue, [], method, exponent, min_exponent)
    print_mov_buf(mov_buf)

    R = RGB_tuple[0]
    G = RGB_tuple[1]
    B = RGB_tuple[2]

    if should_full_crop:
        crops = [auto_crop(B), auto_crop(G), auto_crop(R)]
        min_crop = min(crops, key=crop_eval)

        # Sets new values for the channels
        B = crop(B, min_crop)
        G = crop(G, min_crop)
        R = crop(R, min_crop)

    final = np.dstack([R, G, B])
    skio.imsave(save_name, final)


# Set the call parsing elements
parser = argparse.ArgumentParser(description="Image recoloring through channel alignment.")

parser.add_argument('InputFile',
                    metavar='InputFile',
                    type=str,
                    help="The name of the input image.")

parser.add_argument('OutputFile',
                    metavar='OutputFile',
                    type=str,
                    help="The name of the output image.")

parser.add_argument('--NCC',
                    dest="method",
                    action='store_const',
                    const=NCC,
                    default=SSD,
                    help='Sets image aligning method to Normalized Cross Correlation. (Default: Sum of Squared Differences)')

parser.add_argument('--naive',
                    dest="naive",
                    action='store_const',
                    const=True, default=False,
                    help='Sets image to be aligned exhaustively instead of optimized with the pyramid method.')

parser.add_argument('--autocrop',
                    dest="should_crop",
                    action='store_const',
                    const=True, default=False,
                    help='Toggles auto-cropping method that enables better performance and black border removal.')

parser.add_argument('--full-autocrop',
                    dest="should_full_crop",
                    action='store_const',
                    const=True, default=False,
                    help='Toggles auto-cropping method for the initial image state and the final image state.')

args = parser.parse_args()

start_time = time.time()

if args.naive:
    exhaust_image(args.InputFile, args.OutputFile, args.method, args.should_crop, args.should_full_crop)
else:
    pyramid_search_align(args.InputFile, args.OutputFile, args.method, args.should_crop, args.should_full_crop)

print("Runtime: %.5s seconds" % (time.time() - start_time))