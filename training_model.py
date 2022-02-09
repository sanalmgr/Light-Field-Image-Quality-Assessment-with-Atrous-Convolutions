import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU') #gpus[0], gpus[1], gpus[2]
##########################################################################################################
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input,Conv2D,ELU,MaxPooling2D,Flatten,Dense,Dropout, AtrousConvolution1D, Reshape, Convolution1D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint

#%%
def get_atrous_block_stream1(reshaped_input):
	stream1_atr_conv1 = AtrousConvolution1D(1024, 3, atrous_rate=6, border_mode='same', name='stream1_atr_conv1')(reshaped_input)
	stream1_atr_2dconv1 = Convolution1D(2, 1, padding='same', name='stream1_atr_2dconv1')(stream1_atr_conv1)
	stream1_atr_elu1 = ELU()(stream1_atr_2dconv1)
	
	stream1_atr_conv2 = AtrousConvolution1D(1024, 3, atrous_rate=12, border_mode='same', name='stream1_atr_conv2')(reshaped_input)
	stream1_atr_2dconv2 = Convolution1D(2, 1, padding='same', name='stream1_atr_2dconv2')(stream1_atr_conv2)
	stream1_atr_elu2 = ELU()(stream1_atr_2dconv2)
	
	stream1_atr_conv3 = AtrousConvolution1D(1024, 3, atrous_rate=18, border_mode='same', name='stream1_atr_conv3')(reshaped_input)
	stream1_atr_2dconv3 = Convolution1D(2, 1, padding='same', name='stream1_atr_2dconv3')(stream1_atr_conv3)
	stream1_atr_elu3 = ELU()(stream1_atr_2dconv3)
	
	stream1_atr_conv4 = AtrousConvolution1D(1024, 3, atrous_rate=24, border_mode='same', name='stream1_atr_conv4')(reshaped_input)
	stream1_atr_2dconv4 = Convolution1D(2, 1, padding='same', name='stream1_atr_2dconv4')(stream1_atr_conv4)
	stream1_atr_elu4 = ELU()(stream1_atr_2dconv4)
	
	return stream1_atr_elu1, stream1_atr_elu2, stream1_atr_elu3, stream1_atr_elu4

def get_atrous_block_stream2(reshaped_input):
	stream2_atr_conv1 = AtrousConvolution1D(1024, 3, atrous_rate=6, border_mode='same', name='stream2_atr_conv1')(reshaped_input)
	stream2_atr_2dconv1 = Convolution1D(2, 1, padding='same', name='stream2_atr_2dconv1')(stream2_atr_conv1)
	stream2_atr_elu1 = ELU()(stream2_atr_2dconv1)
	
	stream2_atr_conv2 = AtrousConvolution1D(1024, 3, atrous_rate=12, border_mode='same', name='stream2_atr_conv2')(reshaped_input)
	stream2_atr_2dconv2 = Convolution1D(2, 1, padding='same', name='stream2_atr_2dconv2')(stream2_atr_conv2)
	stream2_atr_elu2 = ELU()(stream2_atr_2dconv2)
	
	stream2_atr_conv3 = AtrousConvolution1D(1024, 3, atrous_rate=18, border_mode='same', name='stream2_atr_conv3')(reshaped_input)
	stream2_atr_2dconv3 = Convolution1D(2, 1, padding='same', name='stream1_atr_2dconv3')(stream2_atr_conv3)
	stream2_atr_elu3 = ELU()(stream2_atr_2dconv3)
	
	stream2_atr_conv4 = AtrousConvolution1D(1024, 3, atrous_rate=24, border_mode='same', name='stream1_atr_conv4')(reshaped_input)
	stream2_atr_2dconv4 = Convolution1D(2, 1, padding='same', name='stream2_atr_2dconv4')(stream2_atr_conv4)
	stream2_atr_elu4 = ELU()(stream2_atr_2dconv4)
	
	return stream2_atr_elu1, stream2_atr_elu2, stream2_atr_elu3, stream2_atr_elu4

def get_model():
	#first stream
	stream1_image=Input(shape=(100, 960, 3))
	#conv1
	stream1_conv1=Conv2D(32, (3, 3), padding='same', name='conv1_stream1')(stream1_image)
	stream1_elu1=ELU()(stream1_conv1)
	stream1_pool1=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_stream1')(stream1_elu1)
	#conv2
	stream1_conv2=Conv2D(32, (3, 3), padding='same', name='conv2_stream1')(stream1_pool1)
	stream1_elu2=ELU()(stream1_conv2)
	stream1_pool2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_stream1')(stream1_elu2)
	#conv3
	stream1_conv3=Conv2D(64, (3, 3), padding='same', name='conv3_stream1')(stream1_pool2)
	stream1_elu3=ELU()(stream1_conv3)
	#conv4
	stream1_conv4=Conv2D(64, (3, 3), padding='same', name='conv4_stream1')(stream1_elu3)
	stream1_elu4=ELU()(stream1_conv4)
	#conv5
	stream1_conv5=Conv2D(128, (3, 3), padding='same', name='conv5_stream1')(stream1_elu4)
	stream1_elu5=ELU()(stream1_conv5)
	stream1_pool5=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='stream1_left')(stream1_elu5)
	#Reshape
	stream1_reshape = Reshape((-1, 128))(stream1_pool5)
	
	#second stream
	stream2_image=Input(shape=(100, 960, 3))
	#conv1
	stream2_conv1=Conv2D(32, (3, 3), padding='same', name='conv1_stream2')(stream2_image)
	stream2_elu1=ELU()(stream2_conv1)
	stream2_pool1=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_stream2')(stream2_elu1)
	#conv2
	stream2_conv2=Conv2D(32, (3, 3), padding='same', name='conv2_stream2')(stream2_pool1)
	stream2_elu2=ELU()(stream2_conv2)
	stream2_pool2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_stream2')(stream2_elu2)
	#conv3
	stream2_conv3=Conv2D(64, (3, 3), padding='same', name='conv3_stream2')(stream2_pool2)
	stream2_elu3=ELU()(stream2_conv3)
	#conv4
	stream2_conv4=Conv2D(64, (3, 3), padding='same', name='conv4_stream2')(stream2_elu3)
	stream2_elu4=ELU()(stream2_conv4)
	#conv5
	stream2_conv5=Conv2D(128, (3, 3), padding='same', name='conv5_stream2')(stream2_elu4)
	stream2_elu5=ELU()(stream2_conv5)
	stream2_pool5=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='stream2_left')(stream2_elu5)
	#Reshape
	stream2_reshape = Reshape((-1, 128))(stream2_pool5)
	
	dns1_stream1, dns2_stream1, dns3_stream1, dns4_stream1 = get_atrous_block_stream1(stream1_reshape)
	dns1_stream2, dns2_stream2, dns3_stream2, dns4_stream2 = get_atrous_block_stream2(stream2_reshape)

	####################################################################################
	#concatenate layerss
	stream1_atr_fusion1 = keras.layers.add([dns1_stream1, dns2_stream1, dns3_stream1, dns4_stream1])
	stream2_atr_fusion1 = keras.layers.add([dns1_stream2, dns2_stream2, dns3_stream2, dns4_stream2])
	fusion3_drop7 = keras.layers.concatenate([stream1_atr_fusion1, stream2_atr_fusion1])
	####################################################################################
	#fc6
	flat6 = Flatten()(fusion3_drop7)
	fc6 = Dense(2560)(flat6)
	elu6 = ELU()(fc6)
	drop6 = Dropout(0.35)(elu6)
	#fc8
	fusion3_fc8=Dense(2560)(drop6)
	#fc9
	predictions=Dense(1)(fusion3_fc8)
	
	model_all=Model(input=[stream1_image, stream2_image], output=predictions, name='all_model')
	model_all.summary()
	
	return model_all

def compile_model(model):
	sgd=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1) #lr=0.0001
	model.compile(loss='mean_squared_error', optimizer=sgd)
	
	return model

def run_model(model, stream1_input, stream2_input, labels):
	# simple early stopping
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
	mc = ModelCheckpoint('model/2stream_mpi_h_v_epis.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

	#fitting model
	history = model.fit(x=[stream1_input, stream2_input], y=[labels], validation_split=0.2, batch_size=128, epochs=6000, verbose=1, callbacks=[es, mc], shuffle=True)
	
	#saving history
	np.save('2stream_mpi_h_v_epis_history.npy',history.history)

#END
