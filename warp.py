import numpy as np
from PIL import Image

def load(im_path):
    im = Image.open(im_path)
    im = np.array(im)
    return im[:,:,:3]

# Get UV inputs to model from UV-textured cube
def warp(im_src):
    C = np.zeros(np.shape(im_src), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            if im_src[i,j,0] !=0 and im_src[i,j,1] !=0:
                u = int(np.floor(im_src[i,j,0]))
                v = int(np.floor(im_src[i,j,1]))
                C[u,v,0] = i
                C[u,v,1] = j
    return C

for i in range(10):
    image = load('./cube_single_UV_train/%05d.png' % i)
    image = warp(image)
    image = Image.fromarray(image)
    image = image.rotate(270)
    image.save('./testingC2/%05d.png' % i)