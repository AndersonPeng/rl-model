import numpy as np
import tensorflow as tf
import ops


class PolicyModel(object):
	#---------------------------
	# Constructor
	#---------------------------
	def __init__(self, sess, ob_space, ac_space, n_env, n_step, n_stack, reuse=False):
		self.sess = sess
		self.init_state = []			#init_state for LSTM
		mb_size = n_env * n_step
		img_height, img_width, channel_dim = ob_space.shape

		#ob_ph: (mb_size, img_height, img_width, channel_dim*n_stack)
		self.ob_ph = tf.placeholder(tf.uint8, (mb_size, img_height, img_width, channel_dim*n_stack))
		ob_normalized = tf.cast(self.ob_ph, tf.float32) / 255.0

		with tf.variable_scope("policy_model", reuse=reuse):
			#conv1: (mb_size, img_height1, img_width1, 32)
			h = ops.conv2d(ob_normalized, 32, 8, 8, 4, 4, name="conv1")
			h = tf.nn.relu(h)
			
			#conv2: (mb_size, img_height2, img_width2, 64)
			h = ops.conv2d(h, 64, 4, 4, 2, 2, name="conv2")
			h = tf.nn.relu(h)
			
			#conv3: (mb_size, img_height3, img_width3, 64)	
			h = ops.conv2d(h, 64, 3, 3, 1, 1, name="conv3")
			h = tf.nn.relu(h)
			
			#fc: (mb_size, 512)
			h = ops.fc(tf.reshape(h, [mb_size, -1]), 512, name="fc1")
			h = tf.nn.relu(h)

			#pi:     (mb_size, ac_space.n)
			#value:  (mb_size, 1)
			pi = ops.fc(h, ac_space.n, name="fc_pi")
			value = ops.fc(h, 1, name="fc_value")
		
		#value:  (mb_size)
		#action: (mb_size)
		self.value = value[:, 0]
		self.action = ops.sample(pi)
		self.pi = pi


	#---------------------------
	# Forward step
	#---------------------------
	def step(self, mb_ob, *_args, **_kwargs):
		a, v = self.sess.run([self.action, self.value], {self.ob_ph: mb_ob})
		return a, v, []


	#---------------------------
	# Forward step for 
	# value function
	#---------------------------
	def value_step(self, mb_ob, *_args, **_kwargs):
		return self.sess.run(self.value, {self.ob_ph: mb_ob})