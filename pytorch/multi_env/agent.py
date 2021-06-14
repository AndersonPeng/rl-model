from utils import linear_lr_decay
import torch
import torch.nn as nn
import numpy as np

#A2C Agent Class
class A2C:
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, policy_net, value_net, lr=1e-4, max_grad_norm=0.5, ent_weight=0.01, device="cuda:0"):
		self.opt_actor     = torch.optim.Adam(policy_net.parameters(), lr)
		self.opt_critic    = torch.optim.Adam(value_net.parameters(), lr)
		self.lr            = lr
		self.max_grad_norm = max_grad_norm
		self.ent_weight    = ent_weight
		self.device        = device

	#-----------------------
	# Train A2C
	#-----------------------
	def train(self, policy_net, value_net, mb_obs, mb_actions, mb_advs, mb_returns):
		mb_obs_tensor       = torch.from_numpy(mb_obs).to(self.device)
		mb_a_logps, mb_ents = policy_net.evaluate(mb_obs_tensor, torch.from_numpy(mb_actions).to(self.device))
		mb_values           = value_net(mb_obs_tensor)

		#A2C loss
		ent     = mb_ents.mean()
		pg_loss = -(torch.from_numpy(mb_advs).to(self.device) * mb_a_logps).mean() - self.ent_weight*ent
		v_loss  = (torch.from_numpy(mb_returns).to(self.device) - mb_values).pow(2).mean()

		#Train actor
		self.opt_actor.zero_grad()
		pg_loss.backward()
		nn.utils.clip_grad_norm_(policy_net.parameters(), self.max_grad_norm)
		self.opt_actor.step()

		#Train critic
		self.opt_critic.zero_grad()
		v_loss.backward()
		nn.utils.clip_grad_norm_(value_net.parameters(), self.max_grad_norm)
		self.opt_critic.step()

		return pg_loss.item(), v_loss.item(), ent.item()

	#-----------------------
	# Learning rate decay
	#-----------------------
	def lr_decay(self, it, n_it):
		linear_lr_decay(self.opt_actor, it, n_it, self.lr)
		linear_lr_decay(self.opt_critic, it, n_it, self.lr)

#PPO Agent Class
class PPO:
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(
		self,
		policy_net,
		value_net,
		lr=1e-4,
		max_grad_norm=0.5,
		ent_weight=0.01,
		clip_val=0.2,
		sample_n_epoch=4,
		sample_mb_size=64,
		mb_size=1024,
		device="cuda:0"
	):
		self.opt_actor      = torch.optim.Adam(policy_net.parameters(), lr)
		self.opt_critic     = torch.optim.Adam(value_net.parameters(), lr)
		self.device         = device
		self.lr             = lr
		self.max_grad_norm  = max_grad_norm
		self.ent_weight     = ent_weight
		self.clip_val       = clip_val
		self.sample_n_epoch = sample_n_epoch
		self.sample_mb_size = sample_mb_size
		self.sample_n_mb    = mb_size // sample_mb_size
		self.rand_idx       = np.arange(mb_size)

	#-----------------------
	# Train PPO
	#-----------------------
	def train(self, policy_net, value_net, mb_obs, mb_actions, mb_old_values, mb_advs, mb_returns, mb_old_a_logps):
		mb_obs         = torch.from_numpy(mb_obs).to(self.device)
		mb_actions     = torch.from_numpy(mb_actions).to(self.device)
		mb_old_values  = torch.from_numpy(mb_old_values).to(self.device)
		mb_advs        = torch.from_numpy(mb_advs).to(self.device)
		mb_returns     = torch.from_numpy(mb_returns).to(self.device)
		mb_old_a_logps = torch.from_numpy(mb_old_a_logps).to(self.device)

		for i in range(self.sample_n_epoch):
			np.random.shuffle(self.rand_idx)

			for j in range(self.sample_n_mb):
				sample_idx         = self.rand_idx[j*self.sample_mb_size : (j+1)*self.sample_mb_size]
				sample_obs         = mb_obs[sample_idx]
				sample_actions     = mb_actions[sample_idx]
				sample_old_values  = mb_old_values[sample_idx]
				sample_advs        = mb_advs[sample_idx]
				sample_returns     = mb_returns[sample_idx]
				sample_old_a_logps = mb_old_a_logps[sample_idx]

				sample_a_logps, sample_ents = policy_net.evaluate(sample_obs, sample_actions)
				sample_values = value_net(sample_obs)
				ent = sample_ents.mean()

				#PPO loss
				v_pred_clip = sample_old_values + torch.clamp(sample_values - sample_old_values, -self.clip_val, self.clip_val)
				v_loss1     = (sample_returns - sample_values).pow(2)
				v_loss2     = (sample_returns - v_pred_clip).pow(2)
				v_loss      = torch.max(v_loss1, v_loss2).mean()

				ratio    = (sample_a_logps - sample_old_a_logps).exp()
				pg_loss1 = -sample_advs * ratio
				pg_loss2 = -sample_advs * torch.clamp(ratio, 1.0-self.clip_val, 1.0+self.clip_val)
				pg_loss  = torch.max(pg_loss1, pg_loss2).mean() - self.ent_weight*ent

				#Train actor
				self.opt_actor.zero_grad()
				pg_loss.backward()
				nn.utils.clip_grad_norm_(policy_net.parameters(), self.max_grad_norm)
				self.opt_actor.step()

				#Train critic
				self.opt_critic.zero_grad()
				v_loss.backward()
				nn.utils.clip_grad_norm_(value_net.parameters(), self.max_grad_norm)
				self.opt_critic.step()

		return pg_loss.item(), v_loss.item(), ent.item()

	#-----------------------
	# Learning rate decay
	#-----------------------
	def lr_decay(self, it, n_it):
		linear_lr_decay(self.opt_actor, it, n_it, self.lr)
		linear_lr_decay(self.opt_critic, it, n_it, self.lr)
