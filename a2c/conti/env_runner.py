import numpy as np


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
	def __init__(self, env, s_dim, a_dim, n_step=5, gamma=0.99):
		self.env = env
		self.n_env = env.n_env
		self.n_step = n_step
		self.gamma = gamma
		self.s_dim = s_dim
		self.a_dim = a_dim

		#obs: (n_env, s_dim)
		self.obs = self.env.reset()


	#--------------------------
	# Get a batch for n steps
	#--------------------------
	def run(self, policy):
		mb_obs, mb_actions, mb_values, mb_rewards, mb_dones, mb_discount_returns = [], [], [], [], [], []

		#1. Run n steps
		#-------------------------------------
		for step in range(self.n_step):
			#actions: (n_env, a_dim)
			#values:  (n_env)
			actions, values = policy.step(self.obs)
			mb_obs.append(np.copy(self.obs))
			mb_values.append(values)
			mb_actions.append(actions)
			
			#obs:     (n_env, s_dim)
			#rewards: (n_env)
			#dones:   (n_env)
			self.obs, rewards, dones, info = self.env.step(actions)
			mb_rewards.append(rewards)
			mb_dones.append(dones)

			for i, done in enumerate(dones):
				if done: self.obs[i] = self.obs[i] * 0

		#last_values: (n_env)
		last_values = policy.value_step(self.obs).tolist()

		#2. Convert to np array & compute returns
		#-------------------------------------
		#mb_obs:     (n_env, n_step, s_dim)
		#mb_actions: (n_env, n_step, a_dim)
		#mb_values:  (n_env, n_step)
		#mb_rewards: (n_env, n_step)
		#mb_dones:   (n_env, n_step)
		mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0)
		mb_actions = np.asarray(mb_actions, dtype=np.float32).swapaxes(1, 0)
		mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
		mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
		mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

		for i, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
			#The last step is not done, add the last value
			if dones[-1] == False:
				mb_discount_returns.append(discount_with_dones(rewards.tolist() + [value], dones.tolist() + [0], self.gamma)[:-1])

			#The last step is done, don't add the last value
			else:
				mb_discount_returns.append(discount_with_dones(rewards.tolist(), dones.tolist(), self.gamma))

		#mb_obs:              (n_env*n_step, s_dim)
		#mb_actions:          (n_env*n_step, a_dim)
		#mb_values:           (n_env*n_step)
		#mb_discount_returns: (n_env*n_step)
		return mb_obs.reshape(self.n_env*self.n_step, self.s_dim), mb_actions.reshape(self.n_env*self.n_step, self.a_dim), \
				mb_values.flatten(), np.asarray(mb_discount_returns, dtype=np.float32).flatten()