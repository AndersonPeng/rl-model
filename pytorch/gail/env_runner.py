import torch
import numpy as np
import gym
from collections import deque


#-----------------------
# Discounted return with dones
#-----------------------
def discount_with_dones(rewards, dones, gamma):
	discounted = []
	r = 0

	for rewards, done in zip(rewards[::-1], dones[::-1]):
		r = rewards + gamma * r * (1. - done)
		discounted.append(r)

	return np.array(discounted[::-1], dtype=np.float32)


#Runner for multiple environment
class EnvRunner:
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, env, s_dim, a_dim, n_step=5, gamma=0.99, device="cuda:0", conti=False):
		self.env    = env
		self.n_env  = env.n_env
		self.n_step = n_step
		self.gamma  = gamma
		self.s_dim  = s_dim
		self.a_dim  = a_dim
		self.device = device
		self.conti  = conti

		#obs: (n_env, s_dim)
		self.obs = self.env.reset()

		#Storages
		self.mb_obs     = np.zeros((self.n_step, self.n_env, self.s_dim), dtype=np.float32)
		self.mb_values  = np.zeros((self.n_step, self.n_env), dtype=np.float32)
		self.mb_rewards = np.zeros((self.n_step, self.n_env), dtype=np.float32)
		self.mb_returns = np.zeros((self.n_env, self.n_step), dtype=np.float32)
		self.mb_a_logps = np.zeros((self.n_step, self.n_env), dtype=np.float32)
		self.mb_dones   = np.zeros((self.n_step, self.n_env), dtype=np.bool)
		self.mb_true_rewards = np.zeros((self.n_step, self.n_env), dtype=np.float32)

		if conti:
			self.mb_actions = np.zeros((self.n_step, self.n_env, self.a_dim), dtype=np.float32)
		else:
			self.mb_actions = np.zeros((self.n_step, self.n_env), dtype=np.int32)

		#Reward & length recorder
		self.total_rewards = np.zeros((self.n_env), dtype=np.float32)
		self.total_len = np.zeros((self.n_env), dtype=np.int32)
		self.reward_buf = deque(maxlen=100)
		self.len_buf = deque(maxlen=100)


	#-----------------------
	# Get a batch for n steps
	#-----------------------
	def run(self, policy_net, value_net, dis_net):
		#1. Run n steps
		#-------------------------------------
		for step in range(self.n_step):
			#obs    : (n_env, s_dim)
			#actions: (n_env) / (n_env, a_dim)
			#a_logps: (n_env)
			#values : (n_env)
			obs_tensor = torch.from_numpy(self.obs).float().to(self.device)
			actions, a_logps = policy_net(obs_tensor)
			values = value_net(obs_tensor)

			actions = actions.cpu().numpy()
			a_logps = a_logps.cpu().numpy()
			values  = values.cpu().numpy()

			self.mb_obs[step, :]     = np.copy(self.obs)
			self.mb_values[step, :]  = values
			self.mb_a_logps[step, :] = a_logps
			self.mb_actions[step, :] = actions
			
			#rewards: (n_env)
			#dones  : (n_env)
			self.obs, rewards, dones, info = self.env.step(actions)
			self.mb_true_rewards[step, :] = rewards
			self.mb_dones[step, :] = dones

		#last_values: (n_env)
		last_values = value_net(torch.from_numpy(self.obs).float().to(self.device))

		#2. Convert to np array
		#-------------------------------------
		#mb_obs:     (n_env, n_step, s_dim)
		#mb_actions: (n_env, n_step) / (n_env, n_step, a_dim)
		#mb_values:  (n_env, n_step)
		#mb_rewards: (n_env, n_step)
		#mb_a_logps: (n_env, n_step)
		#mb_dones:   (n_env, n_step)
		mb_obs          = self.mb_obs.swapaxes(1, 0)
		mb_actions      = self.mb_actions.swapaxes(1, 0)
		mb_values       = self.mb_values.swapaxes(1, 0)
		mb_true_rewards = self.mb_true_rewards.swapaxes(1, 0)
		mb_a_logps      = self.mb_a_logps.swapaxes(1, 0)
		mb_dones        = self.mb_dones.swapaxes(1, 0)

		self.record(mb_true_rewards, mb_dones)

		#3. Compute reward from discriminator
		#-------------------------------------
		#Continuous: concat (s, a)
		if self.conti:
			mb_sa = torch.from_numpy(np.concatenate([
				mb_obs.reshape(self.n_env*self.n_step, -1),
				mb_actions.reshape(self.n_env*self.n_step, -1)
			], 1)).float().to(self.device)
		
		#Discrete: concat (s, a_onehot)
		else:
			mb_actions_onehot = np.zeros([self.n_step*self.n_env, self.a_dim])
			for i in range(self.n_env):
				for j in range(self.n_step):
					mb_actions_onehot[i*self.n_step + j, mb_actions[i, j]] = 1

			mb_sa = torch.from_numpy(np.concatenate([
				mb_obs.reshape(self.n_env*self.n_step, -1),
				mb_actions_onehot
			], 1)).float().to(self.device)

		mb_rewards = -np.log(1e-8 + 1.0 - dis_net(mb_sa).cpu().numpy()).reshape(self.n_env, self.n_step)

		#4. Compute returns
		#-------------------------------------
		for step, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
			#The last step is not done, add the last value
			if dones[-1] == False:
				self.mb_returns[step, :] = discount_with_dones(rewards.tolist() + [value], dones.tolist() + [0], self.gamma)[:-1]

			#The last step is done, don't add the last value
			else:
				self.mb_returns[step, :] = discount_with_dones(rewards.tolist(), dones.tolist(), self.gamma)

		#mb_obs    : (n_env*n_step, s_dim)
		#mb_actions: (n_env*n_step) / (n_env*n_step, a_dim)
		#mb_values : (n_env*n_step)
		#mb_returns: (n_env*n_step)
		#mb_a_logps: (n_env*n_step)
		if self.conti:
			return mb_obs.reshape(self.n_env*self.n_step, self.s_dim), \
					mb_actions.reshape(self.n_env*self.n_step, self.a_dim), \
					mb_values.flatten(), \
					self.mb_returns.flatten(), \
					mb_a_logps.flatten()

		return mb_obs.reshape(self.n_env*self.n_step, self.s_dim), \
				mb_actions.flatten(), \
				mb_values.flatten(), \
				self.mb_returns.flatten(), \
				mb_a_logps.flatten()


	#-----------------------
	# Record reward & length
	#-----------------------
	def record(self, mb_rewards, mb_dones):
		for i in range(self.n_env):
			for j in range(self.n_step):
				if mb_dones[i, j]:
					self.reward_buf.append(self.total_rewards[i])
					self.len_buf.append(self.total_len[i])
					self.total_rewards[i] = mb_rewards[i, j]
					self.total_len[i] = 1
				else:
					self.total_rewards[i] += mb_rewards[i, j]
					self.total_len[i] += 1


	#-----------------------
	# Get performance
	#-----------------------
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