import tensorflow as tf
import ops


class PolicyModel(object):
	#---------------------------
	# Constructor
	#---------------------------
	def __init__(self, sess, img_height, img_width, c_dim, a_dim, reuse=False):
		self.sess = sess

		with tf.variable_scope("policy", reuse=reuse):
			#ob_ph: (mb_size, img_height, img_width, c_dim)
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