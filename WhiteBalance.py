import matplotlib.pyplot as plt
from skimage.io import imread
import skimage.io as skio
img = imread('workshop_pyramid_NCC.jpg')
img_wb = ((img * (img.mean() / img.mean(axis=(0, 1))))
             .clip(0, 255).astype(int))
skio.imsave('workshop_pyramid_NCC_WB.jpg', img_wb)