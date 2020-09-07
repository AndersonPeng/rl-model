import sys
sys.path.insert(0, "../..")

from utils import ops
import tensorflow as tf
import numpy as np


class DiscriminatorModel(object):
	#---------------------------
	# Constructor
	#---------------------------
	def __init__(self, sess, s_dim, a_dim, name="discriminator"):
		self.sess = sess
		self.name = name

		#traj_fake_ph: (mb_size, s_dim+a_dim)
		#traj_real_ph: (mb_size, s_dim+a_dim)
		self.traj_fake_ph = tf.placeholder(tf.float32, [None, s_dim+a_dim], name="trajectory_fake")
		self.traj_real_ph = tf.placeholder(tf.float32, [None, s_dim+a_dim], name="trajectory_real")

		logit_fake, prob_fake = self.build_model(self.traj_fake_ph)
		logit_real, prob_real = self.build_model(self.traj_real_ph, reuse=True)

		#logit_fake: (mb_size)
		#logit_real: (mb_size)
		#prob_fake:  (mb_size)
		#prob_real:  (mb_size)
		self.logit_fake = logit_fake[:, 0]
		self.logit_real = logit_real[:, 0]
		self.prob_fake = prob_fake[:, 0]
		self.prob_real = prob_real[:, 0]


	#---------------------------
	# Build the model
	#---------------------------
	def build_model(self, traj, reuse=False):
		with tf.variable_scope(self.name, reuse=reuse):
			#fc1: (mb_size, 128)
			h = ops.fc(traj, 128, name="dis_fc1")
			h = tf.nn.leaky_relu(h)
				
			#fc2: (mb_size, 128)
			h = ops.fc(h, 128, name="dis_fc2")
			h = tf.nn.leaky_relu(h)

			#fc3: (mb_size, 128)
			h = ops.fc(h, 128, name="dis_fc3")
			h = tf.nn.leaky_relu(h)

			#logit: (mb_size, 1)
			#prob:  (mb_size, 1)
			logit = ops.fc(h, 1, name="dis_fc_logit")
			prob = tf.nn.sigmoid(logit)

		return logit, prob


	#---------------------------
	# Forward step
	#---------------------------
	def step(self, traj_fake):
		return self.sess.run(self.prob_fake, feed_dict={self.traj_fake_ph: traj_fake})