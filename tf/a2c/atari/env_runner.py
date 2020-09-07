import numpy as np
from collections import deque


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


#Runner for multiple environment
class MultiEnvRunner:
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(self, env, img_height, img_width, c_dim, n_step=5, n_stack=4, gamma=0.99):
		self.env = env
		self.n_env = env.n_env
		self.n_step = n_step
		self.n_stack = n_stack
		self.gamma = gamma
		self.img_height = img_height
		self.img_width = img_width
		self.c_dim = c_dim

		#obs: (n_env, img_height, img_width, c_dim*n_stack)
		self.stacked_obs = np.zeros((self.n_env, img_height, img_width, c_dim*n_stack), dtype=np.uint8)
		self.update_stacked_obs(self.env.reset())

		#Reward & length recorder
		self.total_rewards = np.zeros((self.n_env), dtype=np.float32)
		self.total_len = np.zeros((self.n_env), dtype=np.int32)
		self.reward_buf = deque(maxlen=100)
		self.len_buf = deque(maxlen=100)


	#--------------------------
	# Update stacked obs
	#--------------------------
	def update_stacked_obs(self, obs):
		#Shift 1 frame in the stack
		#Then put 1 new frame into the stack
		self.stacked_obs = np.roll(self.stacked_obs, shift=-self.c_dim, axis=3)
		self.stacked_obs[:, :, :, -self.c_dim:] = obs


	#--------------------------
	# Get a batch for n steps
	#--------------------------
	def run(self, policy):
		mb_obs, mb_actions, mb_values, mb_rewards, mb_dones, mb_discount_returns = [], [], [], [], [], []

		#1. Run n steps
		#-------------------------------------
		for step in range(self.n_step):
			#actions: (n_env)
			#values:  (n_env)
			actions, values = policy.step(self.stacked_obs)
			mb_obs.append(np.copy(self.stacked_obs))
			mb_values.append(values)
			mb_actions.append(actions)
			
			#obs:     (n_env, img_height, img_width, c_dim)
			#rewards: (n_env)
			#dones:   (n_env)
			obs, rewards, dones, info = self.env.step(actions)
			mb_rewards.append(rewards)
			mb_dones.append(dones)

			for i, done in enumerate(dones):
				if done: self.stacked_obs[i] = self.stacked_obs[i] * 0

			self.update_stacked_obs(obs)

		#last_values: (n_env)
		last_values = policy.value_step(self.stacked_obs).tolist()

		#2. Convert to np array
		#-------------------------------------
		#mb_obs:     (n_env, n_step, img_height, img_width, c_dim*n_stack)
		#mb_actions: (n_env, n_step)
		#mb_values:  (n_env, n_step)
		#mb_rewards: (n_env, n_step)
		#mb_dones:   (n_env, n_step)
		mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0)
		mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
		mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
		mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
		mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

		self.record(mb_rewards, mb_dones)

		#3. Compute returns
		#-------------------------------------
		for i, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
			#The last step is not done, add the last value
			if dones[-1] == False:
				mb_discount_returns.append(discount_with_dones(rewards.tolist() + [value], dones.tolist() + [0], self.gamma)[:-1])

			#The last step is done, don't add the last value
			else:
				mb_discount_returns.append(discount_with_dones(rewards.tolist(), dones.tolist(), self.gamma))

		#mb_obs:              (n_env*n_step, img_height, img_width, c_dim*n_stack)
		#mb_actions:          (n_env*n_step)
		#mb_values:           (n_env*n_step)
		#mb_discount_returns: (n_env*n_step)
		return mb_obs.reshape(self.n_env*self.n_step, self.img_height, self.img_width, self.c_dim*self.n_stack), \
				mb_actions.flatten(), mb_values.flatten(), np.asarray(mb_discount_returns, dtype=np.float32).flatten()


	#--------------------------
	# Record reward & length
	#--------------------------
	def record(self, mb_rewards, mb_dones):
		for i in range(self.n_env):
			for j in range(self.n_step):
				if mb_dones[i, j] == True:
					self.reward_buf.append(self.total_rewards[i])
					self.len_buf.append(self.total_len[i])
					self.total_rewards[i] = mb_rewards[i, j]
					self.total_len[i] = 1
				else:
					self.total_rewards[i] += mb_rewards[i, j]
					self.total_len[i] += 1

	#--------------------------
	# Get performance
	#--------------------------
	def get_performance(self):
		if len(self.reward_buf) == 0:
			mean_return = 0
			std_return  = 0
		else:
			mean_return = np.mean(self.reward_buf)
			std_return  = np.std(self.reward_buf)

		if len(self.len_buf) == 0:
			mean_len = 0
		else:
			mean_len = np.mean(self.len_buf)

		return mean_return, std_return, mean_len