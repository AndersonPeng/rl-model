import tensorflow as tf
import numpy as np


#Categorical Distribution
class CategoricalDistrib():
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(self, logits):
		self.logits = logits


	#--------------------------
	# Negative log prob
	#--------------------------
	def neg_logp(self, x):
		one_hot_x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
		return tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=one_hot_x)


	#--------------------------
	# Entropy
	#--------------------------
	def entropy(self):
		a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
		ea0 = tf.exp(a0)
		z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
		p0 = ea0 / z0
		return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)


	#--------------------------
	# Sample
	#--------------------------
	def sample(self):
		u = tf.random_uniform(tf.shape(self.logits))
		return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)


#Diagonal Gaussian Distribution
class DiagGaussianDistrib():
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(self, mean, logstd):
		self.mean = mean
		self.logstd = logstd
		self.std = tf.exp(self.logstd)


	#--------------------------
	# Negative log prob
	#--------------------------
	def neg_logp(self, x):
		return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
				+ 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
				+ tf.reduce_sum(self.logstd, axis=-1)


	#--------------------------
	# Entropy
	#--------------------------
	def entropy(self):
		return tf.reduce_sum(self.logstd + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)


	#--------------------------
	# Sample
	#--------------------------
	def sample(self):
		return self.mean + self.std * tf.random_normal(tf.shape(self.mean))