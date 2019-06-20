import numpy as np
from collections import deque


#Runner for multiple environment
class MultiEnvRunner:
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(self, env, s_dim, a_dim, n_step=5, gamma=0.99, lamb=0.95):
		self.env = env
		self.n_env = env.n_env
		self.n_step = n_step
		self.gamma = gamma
		self.lamb = lamb
		self.s_dim = s_dim
		self.a_dim = a_dim

		#obs:   (n_env, s_dim)
		#dones: (n_env)
		self.obs = np.zeros((self.n_env, self.s_dim), dtype=np.float32)
		self.obs[:] = self.env.reset()
		self.dones = [False for _ in range(self.n_env)]

		#Reward & length recorder
		self.total_rewards = np.zeros((self.n_env), dtype=np.float32)
		self.total_len = np.zeros((self.n_env), dtype=np.int32)
		self.reward_buf = deque(maxlen=100)
		self.len_buf = deque(maxlen=100)


	#--------------------------
	# Get a batch for n steps
	#--------------------------
	def run(self, policy, dis):
		mb_obs = []
		mb_actions = [] 
		mb_values = []
		mb_rewards = []
		mb_true_rewards = []
		mb_dones = []
		mb_neg_logprobs = []

		#1. Run n steps
		#-------------------------------------
		for step in range(self.n_step):
			#obs:          (n_env, s_dim)
			#actions:      (n_env)
			#neg_logprobs: (n_env)
			#values:       (n_env)
			actions, values, neg_logprobs = policy.step(self.obs)
			mb_obs.append(np.copy(self.obs))
			mb_actions.append(actions)
			mb_values.append(values)
			mb_neg_logprobs.append(neg_logprobs)
			mb_dones.append(self.dones)

			#rewards: (n_env)
			#dones:   (n_env)
			self.obs[:], true_rewards, self.dones, infos = self.env.step(actions)
			mb_true_rewards.append(true_rewards)

		#last_values: (n_env)
		last_values = policy.value_step(self.obs)

		#2. Convert to np array
		#-------------------------------------
		#mb_obs:          (n_step, n_env, s_dim)
		#mb_actions:      (n_step, n_env)
		#mb_neg_logprobs: (n_step, n_env)
		#mb_values:       (n_step, n_env)
		#mb_rewards:      (n_step, n_env)
		#mb_dones:        (n_step, n_env)
		mb_obs = np.asarray(mb_obs, dtype=np.float32)
		mb_actions = np.asarray(mb_actions, dtype=np.int32)
		mb_values = np.asarray(mb_values, dtype=np.float32)
		mb_dones = np.asarray(mb_dones, dtype=np.bool)
		mb_neg_logprobs = np.asarray(mb_neg_logprobs, dtype=np.float32)
		mb_true_rewards = np.asarray(mb_true_rewards, dtype=np.float32)

		mb_actions_onehot = np.zeros([self.n_step*self.n_env, self.a_dim])
		for j in range(self.n_step):
			for k in range(self.n_env):
				mb_actions_onehot[j*self.n_env + k, mb_actions[j, k]] = 1

		mb_rewards = -np.log(1e-8 + 1. - dis.step(np.concatenate([
			mb_obs.reshape(self.n_step*self.n_env, -1), 
			mb_actions_onehot
		], 1))).reshape(self.n_step, self.n_env)

		self.record(mb_true_rewards, mb_dones)

		#3. Compute returns
		#-------------------------------------
		mb_returns = np.zeros_like(mb_rewards)
		mb_advs = np.zeros_like(mb_rewards)
		last_gae_lam = 0

		for t in reversed(range(self.n_step)):
			if t == self.n_step - 1:
				next_nonterminal = 1.0 - self.dones
				next_values = last_values
			else:
				next_nonterminal = 1.0 - mb_dones[t+1]
				next_values = mb_values[t+1]

			delta = mb_rewards[t] + self.gamma*next_values*next_nonterminal - mb_values[t]
			mb_advs[t] = last_gae_lam = delta + self.gamma*self.lamb*next_nonterminal*last_gae_lam

		mb_returns = mb_advs + mb_values

		#mb_obs:          (n_env*n_step, s_dim)
		#mb_actions:      (n_env*n_step)
		#mb_neg_logprobs: (n_env*n_step)
		#mb_values:       (n_env*n_step)
		#mb_returns:      (n_env*n_step)
		return mb_obs.swapaxes(0, 1).reshape(self.n_env*self.n_step, self.s_dim), \
				mb_actions.swapaxes(0, 1).flatten(), \
				mb_neg_logprobs.swapaxes(0, 1).flatten(), \
				mb_values.swapaxes(0, 1).flatten(), \
				mb_returns.swapaxes(0, 1).flatten()


	#--------------------------
	# Record reward & length
	#--------------------------
	def record(self, mb_rewards, mb_dones):
		for i in range(self.n_env):
			for j in range(self.n_step):
				if mb_dones[j, i] == True:
					self.reward_buf.append(self.total_rewards[i])
					self.len_buf.append(self.total_len[i])
					self.total_rewards[i] = mb_rewards[j, i]
					self.total_len[i] = 1
				else:
					self.total_rewards[i] += mb_rewards[j, i]
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