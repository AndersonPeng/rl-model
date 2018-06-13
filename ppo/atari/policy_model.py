import tensorflow as tf
import numpy as np
import ops
import distribs


class PolicyModel(object):
	#---------------------------
	# Constructor
	#---------------------------
	def __init__(self, sess, img_height, img_width, c_dim, a_dim, name="policy", reuse=False):
		self.sess = sess

		with tf.variable_scope(name, reuse=reuse):
			#ob_ph: (mb_size, s_dim)
			self.ob_ph = tf.placeholder(tf.uint8, [None, img_height, img_width, c_dim], name="observation")
			ob_normalized = tf.cast(self.ob_ph, tf.float32) / 255.0

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
			h = ops.fc(tf.reshape(h, [-1, h.shape[1]*h.shape[2]*h.shape[3]]), 512, name="fc1")
			h = tf.nn.relu(h)

			with tf.variable_scope("actor", reuse=reuse):
				#fc_logits: (mb_size, a_dim)
				logits = ops.fc(h, a_dim, name="a_fc_logits")

			with tf.variable_scope("critic", reuse=reuse):
				#value: (mb_size, 1)
				value = ops.fc(h, 1, name="c_fc_value")

		#value:       (mb_size)
		#action:      (mb_size)
		#neg_logprob: (mb_size)
		self.value = value[:, 0]
		self.distrib = distribs.CategoricalDistrib(logits)
		self.action = self.distrib.sample()
		self.neg_logprob = self.distrib.neg_logp(self.action)


	#---------------------------
	# Forward step
	#---------------------------
	def step(self, mb_obs):
		a, v, nlp = self.sess.run([self.action, self.value, self.neg_logprob], {self.ob_ph: mb_obs})
		return a, v, nlp


	#---------------------------
	# Forward step for 
	# value function
	#---------------------------
	def value_step(self, mb_obs):
		return self.sess.run(self.value, {self.ob_ph: mb_obs})


	#---------------------------
	# Forward step for 
	# sampling actions
	#---------------------------
	def action_step(self, mb_obs):
		return self.sess.run(self.action, {self.ob_ph: mb_obs})