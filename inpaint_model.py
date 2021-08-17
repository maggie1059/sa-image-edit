import logging

import cv2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten
import inpaint_funcs as inpaint

class InpaintCAModel(tf.keras.Model):
    def __init__(self):
        super(InpaintCAModel, self).__init__()

        self.cnum = 48
        self.padding = 'SAME'
        self.training = True
        self.learning_rate = 1e-4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5, beta_2=0.999)

        self.GC1 = inpaint.Gen_Conv(self.cnum, 5, 1, training=self.training, padding=self.padding)
        self.GC2 = inpaint.Gen_Conv(2*self.cnum, 3, 2, training=self.training, padding=self.padding)
        self.GC3 = inpaint.Gen_Conv(2*self.cnum, 3, 1, training=self.training, padding=self.padding)
        self.GC4 = inpaint.Gen_Conv(4*self.cnum, 3, 2, training=self.training, padding=self.padding)
        self.GC5 = inpaint.Gen_Conv(4*self.cnum, 3, 1, training=self.training, padding=self.padding)
        self.GC6 = inpaint.Gen_Conv(4*self.cnum, 3, 1, training=self.training, padding=self.padding)
        self.GC7 = inpaint.Gen_Conv(4*self.cnum, 3, rate=2, training=self.training, padding=self.padding)
        self.GC8 = inpaint.Gen_Conv(4*self.cnum, 3, rate=4, training=self.training, padding=self.padding)
        self.GC9 = inpaint.Gen_Conv(4*self.cnum, 3, rate=8, training=self.training, padding=self.padding)
        self.GC10 = inpaint.Gen_Conv(4*self.cnum, 3, rate=16, training=self.training, padding=self.padding)
        self.GC11 = inpaint.Gen_Conv(4*self.cnum, 3, 1, training=self.training, padding=self.padding)
        self.GC12 = inpaint.Gen_Conv(4*self.cnum, 3, 1, training=self.training, padding=self.padding)
        self.DC1 = inpaint.Gen_Deconv(2*self.cnum, training=self.training, padding=self.padding)
        self.GC13 = inpaint.Gen_Conv(2*self.cnum, 3, 1, training=self.training, padding=self.padding)
        self.DC2 = inpaint.Gen_Deconv(self.cnum, training=self.training, padding=self.padding)
        self.GC14 = inpaint.Gen_Conv(self.cnum//2, 3, 1, training=self.training, padding=self.padding)
        self.GC15 = inpaint.Gen_Conv(3, 3, 1, activation=None, training=self.training, padding=self.padding)

    @tf.function
    def call(self, x):
        x = self.GC1.call(x)
        x = self.GC2.call(x)
        x = self.GC3.call(x)
        x = self.GC4.call(x)
        x = self.GC5.call(x)
        x = self.GC6.call(x)

        x = self.GC7.call(x)
        x = self.GC8.call(x)
        x = self.GC9.call(x)
        x = self.GC10.call(x)
        x = self.GC11.call(x)
        x = self.GC12.call(x)
        x = self.DC1.call(x)
        x = self.GC13.call(x)
        x = self.DC2.call(x)
        x = self.GC14.call(x)
        x = self.GC15.call(x)

        x = tf.nn.tanh(x)
        return x