import sys
sys.path.insert(0, "../..")

from utils import ops, distribs
import tensorflow as tf
import numpy as np


class PolicyModel(object):
	#---------------------------
	# Constructor
	#---------------------------
	def __init__(self, sess, s_dim, a_dim, a_low, a_high, name="policy", reuse=False):
		self.sess = sess

		with tf.variable_scope(name, reuse=reuse):
			#ob_ph: (mb_size, s_dim)
			self.ob_ph = tf.placeholder(tf.float32, [None, s_dim], name="observation")
			self.logstd_ph = tf.placeholder(tf.float32, [1, a_dim], name="logstd")

			with tf.variable_scope("actor", reuse=reuse):
				#fc1: (mb_size, 64)
				h = ops.fc(self.ob_ph, 64, name="a_fc1")
				h = tf.nn.tanh(h)
				
				#fc2: (mb_size, 64)
				h = ops.fc(h, 64, name="a_fc2")
				h = tf.nn.tanh(h)

				#fc_mean (mb_size, a_dim)
				mean = ops.fc(h, a_dim, name="a_fc_mean")

			with tf.variable_scope("critic", reuse=reuse):
				#fc1: (mb_size, 64)
				h = ops.fc(self.ob_ph, 64, name="c_fc1")
				h = tf.nn.tanh(h)
				
				#fc2: (mb_size, 64)
				h = ops.fc(h, 64, name="c_fc2")
				h = tf.nn.tanh(h)

				#value: (mb_size, 1)
				value = ops.fc(h, 1, name="c_fc_value")

		#value:       (mb_size)
		#action:      (mb_size, a_dim)
		#neg_logprob: (mb_size)
		self.value = value[:, 0]
		self.distrib = distribs.DiagGaussianDistrib(mean, self.logstd_ph)
		self.action = self.distrib.sample()
		self.neg_logprob = self.distrib.neg_logp(self.action)


	#---------------------------
	# Forward step
	#---------------------------
	def step(self, mb_obs, logstd):
		a, v, nlp = self.sess.run([self.action, self.value, self.neg_logprob], {
			self.ob_ph: mb_obs,
			self.logstd_ph: logstd
		})
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
	def action_step(self, mb_obs, logstd):
		return self.sess.run(self.action, {
			self.ob_ph: mb_obs,
			self.logstd_ph: logstd
		})