import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import inpaint_helpers

# Identity loss: makes sure original pixel colors were preserved
def identity_loss(xorig, x1, name='identity_loss'):
	b = 1
	h = 256
	w = 256
	c = (tf.slice(xorig, [0,0,0,0], [b, h, w, 2]) +1.)/2.
	x1 = (tf.slice(x1, [0,0,0,0], [b, h, w, 2]) +1.)/2.
	c_prime = (tf.slice(xorig, [0,0,0,2], [b, h, w, 1]))
	num = c_prime * (c - x1)
	denom = tf.math.reduce_sum(tf.abs(c_prime))
	num = tf.math.reduce_sum(tf.abs(num))
	return num/denom

# Reconstruction loss: pixel difference between ground truth and inferred image
def reconstruction_loss(xorig, x1, src_rgb, tgt_uv, tgt_rgb):
	T = make_T(src_rgb, x1)
	W = make_W(tgt_uv, T)

	diff = tgt_rgb[0] - W[0]
	loss = tf.math.reduce_sum(tf.abs(diff))
	return T, W, loss

# Create inpainted texture using original image and inpainted UVs
def make_T(src_rgb, inpainted_tex):
	tex = tf.slice(inpainted_tex, [0,0,0,0], [1,256,256,2])
	tex = (tex+1.)/2.
	tex = tex * 255

	tex_y = tf.slice(tex, [0,0,0,0], [1,256,256,1])
	tex_x = tf.slice(tex, [0,0,0,1], [1,256,256,1])
	swapped_tex = tf.concat([tex_x, tex_y], -1)

	T = tfa.image.resampler(src_rgb, swapped_tex)
	return T

# Create inferred image using inpainted texture T and target pose UVs
def make_W(tgt_uv, T):
	tgt_uv_slice = tf.slice(tgt_uv, [0,0,0,0], [1,256,256,2])
	tgt_uv_slice = tgt_uv_slice * 255

	tex_y = tf.slice(tgt_uv_slice, [0,0,0,0], [1,256,256,1])
	tex_x = tf.slice(tgt_uv_slice, [0,0,0,1], [1,256,256,1])
	swapped_tex = tf.concat([tex_x, tex_y], -1)
	T2 = tf.image.rot90(T)

	W = tfa.image.resampler(T2, swapped_tex)
	return W

def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
		   func=tf.image.ResizeMethod.BILINEAR):
	if dynamic:
		xs = tf.cast(tf.shape(x), tf.float32)
		new_xs = [tf.cast(xs[1]*scale, tf.int32),
				  tf.cast(xs[2]*scale, tf.int32)]
	else:
		xs = tf.shape(x)
		new_xs = [tf.cast((xs[1]*scale), tf.int32), tf.cast((xs[2]*scale), tf.int32)]
	if to_shape is None:
		x = tf.image.resize(x, new_xs, method=func)
	else:
		x = tf.image.resize(x, [to_shape[0], to_shape[1]], method=func)
	return x

# Custom convolution/deconvolution layers for model using gated convolution
class Gen_Conv(tf.keras.layers.Layer):
	def __init__(self, cnum, ksize, stride=1, rate=1, name='conv',
			 padding='SAME', activation=tf.nn.elu, training=True):
		super(Gen_Conv, self).__init__()

		self.xf = tf.keras.layers.Conv2D(cnum, ksize, stride, dilation_rate=rate, activation=None, padding=padding, name=name) 
		self.xg = tf.keras.layers.Conv2D(cnum, ksize, stride, dilation_rate=rate, activation=None, padding=padding)
		self.bn = tf.keras.layers.BatchNormalization()
		self.activation = activation
		self.cnum = cnum

	def call(self, inputs):
		x_out = self.xf(inputs)
		# gating
		y_out = self.xg(inputs)

		if self.cnum == 3 or self.activation is None:
			# conv for output
			return x_out
		
		x_out = self.activation(x_out)
		y_out = tf.nn.sigmoid(y_out)
		x_out = x_out * y_out
		x_out = self.bn(x_out)
		return x_out

class Gen_Deconv(tf.keras.layers.Layer):
	def __init__(self, cnum, name='upsample', padding='SAME', training=True):
		super(Gen_Deconv, self).__init__()

		self.cnum = cnum
		self.padding = padding
		self.conv = Gen_Conv(cnum, 3, 1, name=name+'_conv', padding=padding,
			training=training)

	def call(self, x):
		x = resize(x, func=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		x = self.conv(x)
		return x