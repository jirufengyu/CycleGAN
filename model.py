import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
import scipy
import numpy as np
import matplotlib.pyplot as plt
class GAN():
    def __init__(self):
        self.img_rows=128
        self.img_cols=128
        self.channels=3
        self.img_shapes=(self.img_rows,self.img_cols,self.channels)
        self.IE_filters=32
    #def T_Encoder(self,x):

    def I_Encoder(self,x=0):
        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            #d = InstanceNormalization()(d)#对单个对象进行正则化
            return d
        d0=Input(shape=self.img_shapes)
        print(d0.shape)
        d1=conv2d(d0,self.IE_filters)
        print(d1.shape)
        d2=conv2d(d1,self.IE_filters*2)
        print(d2.shape)
        d3 = conv2d(d2, self.IE_filters * 4)
        print(d3.shape)
        d4 = conv2d(d3, self.IE_filters * 8)
        print(d4.shape)
        return Model(d0,d4)
    def I_Encoder(self,x=0):
        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            #u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])

        u1 = deconv2d(d4, d3, self.gf * 4)
        u2 = deconv2d(u1, d2, self.gf * 2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)



g=GAN()
img=plt.imread("img.png").astype(np.float)
print(img.shape)
m=g.I_Encoder(img)

