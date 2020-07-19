#
# Activation functions used for deep complex neural network
#
# Author: Jianping Wang

import numpy as np 
from tensorflow.keras import backend as K
import math
import tensorflow as tf 


def modReLU(z, axis=-1):
	"""modReLU activation function for deep complex neural network
		modeReLU(z) = relu(|z| + b) * z / |z|, where b defines the 
		"dead zone" in the complex plane. In the case where b>=0,
		the whole complex plane would preserve both amplitude and 
		phase information.
	#Argument
		z:     complex-valued input tensor
		axis:  the axis of the channel.	 
	"""
	ndim = K.ndim(z)
    input_shape = K.shape(z)     # Channel dimension
    input_shape_array = input_shape[1:]  

    with tf.name_scope("bias") as scope:
        bias = tf.get_variable(name=scope,
                               shape=input_shape_array,
                               initializer=tf.constant_initializer(0.0),
                               trainable=True)
    # relu(|z|+b) * (z / |z|)
    norm = K.abs(z)
    scale = K.relu(norm + bias) / (norm + 1e-6)
    output = tf.dtypes.complex(tf.math.real(z) * scale,
                        tf.math.imag(z) * scale)
    return output



def zReLU(z)
	""" zReLU activation function, which perserves the amplitude and
		phase information when z is located in the first quadrant.
		zReLU(z) = z if 0 <= theta_z <= pi/2, otherwise 0
	#Argument
		z: complex-valued input tensor 
	"""
  	# Compute the phase of input complex number
  	phase = tf.math.angle(z)
	# Check whether phase <= pi/2
  	le = tf.math.less_equal(phase, pi / 2)
	# if phase <= pi/2, keep it in comp
	# if phase > pi/2, throw it away and set comp equal to 0
  	y = tf.zeros_like(z)
  	z = tf.where(le, z, y)
  	# Check whether phase >= 0
  	ge = tf.math.greater_equal(phase, 0)
  	# if phase >= 0, keep it
  	# if phase < 0, set output equal to 0
  	output = tf.where(ge, z, y)
  	return output



def complex_to_channels(z, name="complex2channels", axis=-1):
    """Convert data from complex to channels.
    # Argument:
      z: complex-valued tensor
    # Output:
        a real-valued tensor with the real and imaginary parts concatenated along the dimension of 'axis' 
    """
    with tf.name_scope(name):
        z_ri = tf.concat([tf.math.real(z), tf.math.imag(z)], axis=axis)
    return z_ri


def channels_to_complex(z, name="channels2complex",axis=-1):
    """Convert data from channels to complex.
    # Argument:
      z: tensor with the real and imaginary parts concatenated along the dimension of 'axis'
    # Output:
         complex-valued tensor
    """
    ndim = K.ndim(z)
    input_dim = K.shape(z)[axis] // 2
    
    if ndim == 2:
      z_real = z[:, :input_dim]
      z_imag = z[:, input_dim:]
    elif axis == 1:
      z_real = z[:, :input_dim, ...]
      z_imag = z[:, input_dim:, ...]
    elif axis == -1:
      z_real = z[..., :input_dim]
      z_imag = z[..., input_dim:]
    else:
      raise ValueError(
        'Incorrect axis or dimension of the tensor. axis should be'
        'either 1 or -1.'
        )

    with tf.name_scope(name):
        complex_out = tf.complex(z_real, z_imag)
    return complex_out  



class Complex2Channel(Layer):
  """docstring for Complex2Channel"""
  def __init__(self, axis=-1, **kwargs):
    super(Complex2Channel, self).__init__(**kwargs)
    self.axis = axis
  
  def call(self, inputs):
    return complex_to_channels(inputs)

  def compute_output_shape(self, input_shape, axis=-1):
    out_shape = list(input_shape)
    out_shape[axis] = out_shape[axis] * 2
    return tuple(out_shape)

class Channel2Complex(Layer):
  """docstring for Channel2Complex"""
  def __init__(self, axis=-1, **kwargs):
    super(Channel2Complex, self).__init__(**kwargs)
    self.axis = axis

  def call(self, inputs):
    return channels_to_complex(inputs)

  def compute_output_shape(self, input_shape, axis=-1):
    out_shape = list(input_shape)
    out_shape[axis] = out_shape[axis] // 2
    return tuple(out_shape)
    


      
    
    
