import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import skimage.io as skio
from skimage import img_as_ubyte
from matplotlib.patches import Rectangle

dinner = imread('workshop_pyramid_NCC.jpg')
plt.imshow(dinner, cmap='gray')
fig, ax = plt.subplots(1,2, figsize=(10,6))
ax[0].imshow(dinner)
ax[0].set_title('Original Image')
dinner_gw = ((dinner * (dinner.mean() / dinner.mean(axis=(0, 1))))
             .clip(0, 255).astype(int))
ax[1].imshow(dinner_gw);
ax[1].set_title('Whitebalanced Image');
plt.show()
skio.imsave('workshop_pyramid_NCC_WB.jpg', dinner_gw)