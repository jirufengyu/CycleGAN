import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU,LSTM
from tensorflow.keras.layers import UpSampling2D, Conv2D,Embedding
from tensorflow.keras.models import Sequential, Model
import scipy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
class GAN():
    def __init__(self):
        self.img_rows=128
        self.img_cols=128
        self.channels=3
        self.img_shapes=(self.img_rows,self.img_cols,self.channels)
        self.IE_filters=32

        self.max_features=10000#词汇数量
        self.maxlen=500#
        self.embedding_dims=256
    #def T_Encoder(self,x):

    def I_Encoder(self,x=0):
        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            #d = InstanceNormalization()(d)#对单个对象进行正则化
            return d
        d0=Input(shape=self.img_shapes)
        d1=conv2d(d0,self.IE_filters)
        d2=conv2d(d1,self.IE_filters*2)
        d3 = conv2d(d2, self.IE_filters * 4)
        d4 = conv2d(d3, self.IE_filters * 8)
        flatten=Flatten()
        d4=flatten(d4)
        print(d4.shape)
        return d4#Model(d0,d4)
    def I_Decoder(self,x=0):
        def deconv2d(layer_input,  filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            #u = InstanceNormalization()(u)
            #u = Concatenate()([u, skip_input])//类似resnet把某些层跳过
            return u

        u1 = deconv2d(x, self.IE_filters * 4)
        u2 = deconv2d(u1,self.IE_filters * 2)
        u3 = deconv2d(u2, self.IE_filters)

        u4 = UpSampling2D(size=2)(u3)

        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return output_img#Model(x, output_img)

    def T_Encoder(self,x=0):
        embedding = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims, input_length=self.maxlen)
        LSTM1 = LSTM(256, return_sequences=True, input_shape=(50, 300))
        drop1 = Dropout(0.5)

        flatten = Flatten()
        dense=Dense(64,activation='relu')

        x=embedding(x)
        print(x.shape)
        x=LSTM1(x)
        print(x.shape)
        x=drop1(x)
        print(x.shape)
        x=flatten(x)
        print(x.shape)
        x=dense(x)
        print(x.shape)
        return x
    #def T_Deconder(self,x):

    def C_Deconder(self,h_img,h_text,c_dim):
        '''

        :param h_img: 输入的图片
        :param h_text:
        :param c_dim: 中心编码器的隐藏层维度
        :return: 解码的总向量，图像向量，文本向量
        '''
        flatten=Flatten()
        i_shape=h_img.shaep
        h_img=flatten(h_img)
        I_inputdim=h_img.shape[1].value
        T_inputdim=h_text.shape[1].value
        h=tf.concat([h_img,h_text],0)
        ori_dim=h.shape[1].value
        dense0=Dense(c_dim,activation='relu')
        h=dense0(h)
        dense1=Dense(ori_dim,activation='relu')
        h=dense1(h)
        I_out,T_out=tf.split(h,[I_inputdim,T_inputdim],1)
        I_out=tf.reshape(I_out,i_shape)
        return h,I_out,T_out

    def T_Deconder(self,x):
        LSTM2=LSTM(256, return_sequences=True, input_shape=(50, 300))
'''
class I_Enconder(layers.Layer):
    def __init__(self,f_sizes=4):
        super().__init__()
        self.f_sizes=f_sizes
        def conv2d(layer_input, filters, f_size=self.f_sizes):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            #d = InstanceNormalization()(d)#对单个对象进行正则化
            return d
        self.
'''








g=GAN()
img=plt.imread("img.png").astype(np.float)
m=g.I_Encoder(img)
#y=g.I_Decoder(m)
x=tf.random.uniform([50,300])
a=g.T_Encoder(x)



