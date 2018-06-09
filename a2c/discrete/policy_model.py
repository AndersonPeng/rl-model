import tensorflow as tf
import ops


class PolicyModel(object):
	#---------------------------
	# Constructor
	#---------------------------
	def __init__(self, sess, s_dim, a_dim, reuse=False):
		self.sess = sess

		#ob_ph: (mb_size, s_dim)
		self.ob_ph = tf.placeholder(tf.float32, (None, s_dim))

		with tf.variable_scope("policy_model", reuse=reuse):
			#conv1: (mb_size, 64)
			h = ops.fc(self.ob_ph, 64, name="fc1")
			h = tf.nn.relu(h)
			
			#conv2: (mb_size, 128)
			h = ops.fc(h, 128, name="fc2")
			h = tf.nn.relu(h)
			
			#conv3: (mb_size, 128)	
			h = ops.fc(h, 128, name="fc3")
			h = tf.nn.relu(h)
			
			#fc: (mb_size, 64)
			h = ops.fc(h, 64, name="fc4")
			h = tf.nn.relu(h)

			#pi:     (mb_size, a_dim)
			#value:  (mb_size, 1)
			pi = ops.fc(h, a_dim, name="fc_pi")
			value = ops.fc(h, 1, name="fc_value")
		
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