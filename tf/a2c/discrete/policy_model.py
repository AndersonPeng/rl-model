import sys
sys.path.insert(0, "../..")

from utils import ops
import tensorflow as tf


class PolicyModel(object):
	#---------------------------
	# Constructor
	#---------------------------
	def __init__(self, sess, s_dim, a_dim, reuse=False):
		self.sess = sess

		with tf.variable_scope("policy", reuse=reuse):
			#ob_ph: (mb_size, s_dim)
			self.ob_ph = tf.placeholder(tf.float32, [None, s_dim], name="observation")

			with tf.variable_scope("actor", reuse=reuse):
				#fc1: (mb_size, 64)
				h = ops.fc(self.ob_ph, 64, name="a_fc1")
				h = tf.nn.relu(h)
				
				#fc2: (mb_size, 128)
				h = ops.fc(h, 128, name="a_fc2")
				h = tf.nn.relu(h)
				
				#fc3: (mb_size, 128)	
				h = ops.fc(h, 128, name="a_fc3")
				h = tf.nn.relu(h)

				#pi:     (mb_size, a_dim)
				pi = ops.fc(h, a_dim, name="a_fc_pi")

			with tf.variable_scope("critic", reuse=reuse):
				#fc1: (mb_size, 64)
				h = ops.fc(self.ob_ph, 64, name="c_fc1")
				h = tf.nn.relu(h)
				
				#fc2: (mb_size, 128)
				h = ops.fc(h, 128, name="c_fc2")
				h = tf.nn.relu(h)
				
				#fc3: (mb_size, 128)	
				h = ops.fc(h, 128, name="c_fc3")
				h = tf.nn.relu(h)

				#value:  (mb_size, 1)
				value = ops.fc(h, 1, name="c_fc_value")
		
		#value:  (mb_size)
		#action: (mb_size)
		self.value = value[:, 0]
		self.cat_dist = tf.distributions.Categorical(pi)
		self.action = self.cat_dist.sample(1)[0]
		self.pi = pi


	#---------------------------
	# Forward step
	#---------------------------
	def step(self, mb_obs):
		a, v = self.sess.run([self.action, self.value], {self.ob_ph: mb_obs})
		return a, v


	#---------------------------
	# Forward step for 
	# value function
	#---------------------------
	def value_step(self, mb_obs):
		return self.sess.run(self.value, {self.ob_ph: mb_obs})