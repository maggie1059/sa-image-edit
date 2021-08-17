import os
import glob
import math
import cv2

import tensorflow as tf
from PIL import Image, ImageDraw

from inpaint_model import InpaintCAModel, Discriminator
from preprocess import load_image_batch, load_test_batch
from inpaint_funcs import identity_loss, reconstruction_loss, make_T, make_W

# Training or testing
TEST = False
# Max number of epochs to train for
NUM_EPOCHS = 1000
# Directory for training data
LOAD_DIR = './mix/mix_train'
# Directory for test data
TEST_DIR = './mix/mix_test'

# Writes out images for inferred results from model
def write_out(x1, batch, T, W, src_rgb, tgt_rgb):
		x1 = tf.slice(x1, [0, 0, 0, 0], [1, 256, 256, 2])
		x1 = (x1+1.)/2.
		zeros = tf.zeros([1, 256, 256, 1])
		x1 = tf.concat([x1, zeros], axis=-1)

		batch = tf.slice(batch, [0, 0, 0, 0], [1, 256, 256, 2])
		batch = (batch+1.)/2.
		batch = tf.concat([batch, zeros], axis=-1)
		
		# Inferred inpainted texture
		x1 = tf.image.convert_image_dtype(x1[0, :, :, :], dtype=tf.uint8)
		out = tf.io.encode_jpeg(x1)
		fwrite = tf.io.write_file(tf.constant('out_test.jpg'), out)

		# Original UVs
		batch = tf.image.convert_image_dtype(batch[0, :, :, :], dtype=tf.uint8)
		out2 = tf.io.encode_jpeg(batch)
		fwrite = tf.io.write_file(tf.constant('out_orig.jpg'), out2)

		# Original textured image of cube
		src_rgb = tf.image.convert_image_dtype(src_rgb[0, :, :, :], dtype=tf.uint8)
		outsrc = tf.io.encode_jpeg(src_rgb)
		fwrite = tf.io.write_file(tf.constant('out_src.jpg'), outsrc)

		# Ground-truth image of cube in target pose
		tgt_rgb = tf.image.convert_image_dtype(tgt_rgb[0, :, :, :], dtype=tf.uint8)
		outtgt = tf.io.encode_jpeg(tgt_rgb)
		fwrite = tf.io.write_file(tf.constant('out_tgt.jpg'), outtgt)

		# Inpainted texture
		T = tf.image.convert_image_dtype(T[0,:, :, :], dtype=tf.uint8)
		outT = tf.io.encode_jpeg(T)
		fwrite = tf.io.write_file(tf.constant('out_T.jpg'), outT)

		# Inferred textured cube in target pose
		W = tf.image.convert_image_dtype(W[0,:, :, :], dtype=tf.uint8)
		outW = tf.io.encode_jpeg(W)
		fwrite = tf.io.write_file(tf.constant('out_W.jpg'), outW)

def train(GAN, dataset_iterator, test_iterator, manager):
	test_objs = list(test_iterator.as_numpy_iterator())
	total_loss = 0
	for iteration, fullbatch in enumerate(dataset_iterator):
		src_rgb = tf.slice(fullbatch, [0,0,0,0], [1, 256, 256, 3])
		batch = tf.slice(fullbatch, [0,0,0,3], [1, 256, 256, 5])
		test_im = test_objs[iteration]
		tgt_rgb = tf.slice(test_im, [0,0,0,0], [1, 256, 256, 3])
		tgt_uv = tf.slice(test_im, [0,0,256,0], [1, 256, 256, 3])

		# Inference and write out images
		if TEST == True and iteration==10: #change iteration number for different example
			x1 = GAN.call(batch)
			T, W, loss = reconstruction_loss(batch, x1, src_rgb, tgt_uv, tgt_rgb)
			write_out(x1, batch, T, W, src_rgb, tgt_rgb)
			exit()

		with tf.GradientTape() as gen_tape:
			x1 = GAN.call(batch)
			id_loss = identity_loss(batch, x1)
			T, W, reconstruct_loss = reconstruction_loss(batch, x1, src_rgb, tgt_uv, tgt_rgb)
			loss = id_loss + (1e-5 * reconstruct_loss)

		gen_grad = gen_tape.gradient(loss, GAN.trainable_variables)
		GAN.optimizer.apply_gradients(zip(gen_grad, GAN.trainable_variables))
		total_loss += loss
	loss_eq = total_loss/1000.0
	print("LOSS: ", loss_eq)

def main():
	dataset_iterator = load_image_batch(LOAD_DIR)
	test_iterator = load_test_batch(TEST_DIR)
	GAN = InpaintCAModel()

	# Directory for model checkpoints
	checkpoint_dir = './checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator=GAN)
	manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

	# Ensure the output directory exists
	if not os.path.exists(OUT_DIR):
			os.makedirs(OUT_DIR)

	try:
		# Restore previous checkpoints
		# checkpoint.restore(manager.latest_checkpoint)
		for epoch in range(0, NUM_EPOCHS):
			print('========================== EPOCH %d  ==========================' % epoch)
			train(GAN, dataset_iterator, test_iterator, manager)
			if epoch % 1 == 0:
				print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
				manager.save()

	except RuntimeError as e:
		print(e)

if __name__ == "__main__":
	main()
