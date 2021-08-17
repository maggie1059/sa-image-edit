import numpy as np
from PIL import Image

# Take largest possible square from image
def square(im):
    dims = im.shape
    width = dims[0]
    height = dims[1]
    length = min(width, height)
    if len(dims) == 3:
        im = im[:length, :length, :3]
    elif len(dims) == 2:
        im = im[:length, :length]
        im = np.expand_dims(im, axis=-1)
        im = np.tile(im, (1,1,3))
    return im

# Cut out cubemap from texture
def cube(im):
    length = im.shape[0] #256
    side = length//4 #64
    mask = np.zeros(im.shape[:2])
    #middle strip
    bottom_center = int(length//2 - side//2) #96
    top_center = length - (4*side) #0
    mask[top_center:, bottom_center:bottom_center+side] = 1 #mask[0:256, 96:160]
    #bottom strip
    bottom_left = bottom_center - side #32
    top_left = length - side #192
    mask[top_left:, bottom_left:bottom_left+(3*side)] = 1 #mask[192:256, 32:224]
    if len(im.shape) == 2:
        im = im * mask
    else:
        for i in range(im.shape[2]):
            im[:,:,i] = im[:,:,i] * mask 
    return im

# For when you want each face to have the same texture image
def repeat_cube(im):
    new_im = np.zeros((256, 256, 3), dtype=np.uint8)
    new_im[0:64, 96:160, :] = im[:,:,:]
    new_im[64:128, 96:160, :] = im[:,:,:]
    new_im[128:192, 96:160, :] = im[:,:,:]
    new_im[192:, 96:160, :] = im[:,:,:]
    new_im[192:, 32:96, :] = im[:,:,:]
    new_im[192:, 160:224, :] = im[:,:,:]
    return new_im

def load_image(image_path):
    im = Image.open(image_path)
    im = np.array(im)
    im = square(im)
    im = Image.fromarray(im)
    im = im.resize((64,64))
    im = np.array(im)
    return im

# Load original image for texture
image = load_image('./textures/pattern1.jpeg')

# Commented out line below for repeating same texture on each face
# image = repeat_cube(image)
image = square(image)
image = cube(image)

image = Image.fromarray(image)
image = image.resize((256,256))
image.save('./new_textures/pattern1.png')