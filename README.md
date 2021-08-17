# Shape Assembly Image Editing

The model in this project is based on the inpainting network from [Coordinate-based Texture Inpainting](https://arxiv.org/pdf/1811.11459.pdf).

The goal of this project is to predict novel views of a textured cube non-parametrically by using existing pixel values from an input image to inpaint missing portions of the cube's texture in a new pose. This is done by inpainting UV coordinates rather than actual color values from the image, which forces the model to grab existing pixels from the known parts of the texture for the inpainting process.

#### Example of model inference for wood texture

Original input image and UVs:
<p float="left">
  <img src="/ex_ims/out_src.jpg" width="35%" />
  <img src="/ex_ims/out_orig.jpg" width="35%" /> 
</p>

Model-inpainted UVs and texture:
<p float="left">
  <img src="/ex_ims/out_test.jpg" width="35%" />
  <img src="/ex_ims/out_T.jpg" width="35%" /> 
</p>

Target image in new pose:
<p float="left">
  <img src="/ex_ims/out_tgt.jpg" width="35%" />
</p>

Model's predicted image of new pose:
<p float="left">
  <img src="/ex_ims/out_W.jpg" width="35%" />
</p>


### Usage Pipeline

#### Dataset Generation

To make a new dataset, use `make_cubes.py` to generate 256x256 images of textured cubes in different poses. The same code can be used to generate UV "textured" cubes, which are required for testing. To create a new texture cubemap using any RGB image, run `cubemap.py` using the desired texture. The resulting cubemap should be used in `make_cubes.py` as the texture image. To create the cubemap UVs used in the training dataset, run `warp.py` on the UV cubes created using `make_cubes.py`.

#### Model Usage

The inpainting model can be found in `inpaint_model.py`. To train the model, run `train.py`. The data directories, number of epochs, and whether or not to train or test the model can be changed at the top of `train.py`. Running inference will output the 6 images shown above for one example input. The `heaptmap.py` script is included for analysis, and running it on an inpainted UV image will produce a heatmap of which/how many original pixels were used to create the inpainted texture.

#### Data Input Requirements

Training data should consist of 512x256 images, created by concatenating 2 256x256 images. The left should be the textured cube in its original pose (as created by `make_cubes.py`), and the right should be the UV cubemap (as created by `warp.py`).

For example:
<p float="left">
  <img src="/repeat/train/1/00733.png" width="35%" />
</p>

Testing data should be in a similar format as the training data, with 2 256x256 images concatenated into one 512x256 image. The left should be the ground truth image of the textured cube in a new pose, and the right should be the UVs of the cube in the target pose.

For example:
<p float="left">
  <img src="/repeat/test/1/00053.png" width="35%" />
</p>


### Data

This repo contains 2 separate datasets: `/repeat` and `/mix`. The `/repeat` dataset uses a single woodgrain texture repeated across all 6 faces of the cube. The `/mix` dataset contains 7 different textures, 5 of which are woodgrain, with 1 marble and 1 patterned texture. To use either of these datasets, change the `LOAD_DIR` and `TEST_DIR` paths at the top of `train.py`.