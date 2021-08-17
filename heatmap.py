import numpy as np
from PIL import Image

def load(im_path):
    im = Image.open(im_path)
    im = np.array(im)
    return im[:,:,:3]

# Make heatmap of which pixels from original image were used and how often in inpainted texture
def make_heatmap():
    # Load inpainted UVs
    uv = load('./out_test.jpg')
    uv = np.array(uv)
    heatmap = np.zeros((256,256,3))
    for i in range(256):
        for j in range(256):
            u = int(np.floor(uv[i,j,0]))
            v = int(np.floor(uv[i,j,1]))
            heatmap[u,v,0] += 10
            heatmap[u,v,1] += 10
            heatmap[u,v,2] += 10
    heatmap = Image.fromarray(heatmap.astype(np.uint8))
    heatmap.save('./heatmap.png')

make_heatmap()