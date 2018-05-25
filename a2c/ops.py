import tensorflow as tf


#-----------------------------------
# Fully connected
#-----------------------------------
def fc(x, out_dim, name="fc", bias_start=0.0):
	shape = x.get_shape().as_list()
	stddev = tf.sqrt(3.0 / (shape[1] + out_dim))

	with tf.variable_scope(name):
		W = tf.get_variable(
			"W",
			[shape[1], out_dim],
			tf.float32,
			initializer=tf.random_normal_initializer(stddev=stddev)
		)

		b = tf.get_variable(
			"b",
			[out_dim],
			initializer=tf.constant_initializer(bias_start)
		)

		return tf.matmul(x, W) + b


#-----------------------------------
# Convolution 2D 
#-----------------------------------
def conv2d(x, out_dim, k_h=5, k_w=5, d_h=2, d_w=2, name="conv2d"):
	shape = x.get_shape().as_list()
	stddev = tf.sqrt(3.0 / (k_h*k_w*shape[-1] + out_dim))

	with tf.variable_scope(name):
		W = tf.get_variable(
			"W", 
			[k_h, k_w, shape[-1], out_dim],
			initializer=tf.truncated_normal_initializer(stddev=stddev)
		)
		b = tf.get_variable(
			"b",
			[out_dim],
			initializer=tf.constant_initializer(0.0)
		)

		conv = tf.nn.conv2d(x, W, strides=[1, d_h, d_w, 1], padding="SAME")
		conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
		return conv


#--------------------------
# Sample from the distrib. 
# (greedy)
#--------------------------
def sample(logits):
	noise = tf.random_uniform(tf.shape(logits))
	return tf.argmax(logits - tf.log(-tf.log(noise)), 1)