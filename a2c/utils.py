import tensorflow as tf
import numpy as np


#-----------------------------------
# Show all trainable variables
#-----------------------------------
def show_all_vars():
	tf.contrib.slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)


#--------------------------
# Cat entropy
#--------------------------
def cat_entropy(logits):
	a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
	ea0 = tf.exp(a0)
	z0 = tf.reduce_sum(ea0, 1, keepdims=True)
	p0 = ea0 / z0

	return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


#--------------------------
# Find trainable variables
#--------------------------
def find_trainable_vars(key):
	with tf.variable_scope(key):
		return tf.trainable_variables()


#--------------------------
# Discounted return with dones
#--------------------------
def discount_with_dones(rewards, dones, gamma):
	discounted = []
	r = 0

	for rewards, done in zip(rewards[::-1], dones[::-1]):
		r = rewards + gamma * r * (1. - done)
		discounted.append(r)

	return discounted[::-1]


#--------------------------
# Explained variance
#--------------------------
def explained_variance(ypred, y):
	assert y.ndim == 1 and ypred.ndim == 1

	vary = np.var(y)
	return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary