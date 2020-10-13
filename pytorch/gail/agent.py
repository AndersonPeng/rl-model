from utils import linear_lr_decay
import torch
import torch.nn as nn
import numpy as np


#A2C Agent Class
class A2C:
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(
		self, 
		policy_net, 
		value_net, 
		dis_net, 
		device, 
		a_dim,
		lr=1e-4, 
		max_grad_norm=0.5, 
		ent_weight=0.01,
		mb_size=128,
		conti=False
	):
		self.opt_actor     = torch.optim.Adam(policy_net.parameters(), lr)
		self.opt_critic    = torch.optim.Adam(value_net.parameters(), lr)
		self.opt_dis       = torch.optim.Adam(dis_net.parameters(), lr)
		self.device        = device
		self.a_dim         = a_dim
		self.lr            = lr
		self.max_grad_norm = max_grad_norm
		self.ent_weight    = ent_weight
		self.criterion     = nn.BCELoss()
		self.ones_label    = torch.autograd.Variable(torch.ones((mb_size, 1))).to(self.device)
		self.zeros_label   = torch.autograd.Variable(torch.zeros((mb_size, 1))).to(self.device)
		self.conti         = conti


	#-----------------------
	# Train A2C
	#-----------------------
	def train(self, policy_net, value_net, dis_net, mb_obs, mb_actions, mb_advs, mb_returns, sa_real):
		mb_obs_tensor       = torch.from_numpy(mb_obs).to(self.device)
		mb_a_logps, mb_ents = policy_net.evaluate(mb_obs_tensor, torch.from_numpy(mb_actions).to(self.device))
		mb_values           = value_net(mb_obs_tensor)

		if self.conti:
			mb_sa_fake = np.concatenate([mb_obs, mb_actions], 1)
		else:
			mb_actions_onehot = np.zeros([mb_actions.shape[0], self.a_dim])

			for i in range(len(sample_actions_onehot)):
				mb_actions_onehot[i, mb_actions[i]] = 1

			mb_sa_fake = np.concatenate([mb_obs, mb_actions_onehot], 1)

		mb_sa_real = sa_real[np.random.randint(0, sa_real.shape[0], len(sample_actions_onehot)), :]
		
		#A2C loss
		ent     = mb_ents.mean()
		pg_loss = -(torch.from_numpy(mb_advs).to(self.device) * mb_a_logps).mean() - self.ent_weight*ent
		v_loss  = (torch.from_numpy(mb_returns).to(self.device) - mb_values).pow(2).mean()
		ac_loss = pg_loss + 0.2*v_loss

		#Adversarial loss
		dis_real = dis_net(torch.from_numpy(mb_sa_real).to(self.device))
		dis_fake = dis_net(torch.from_numpy(mb_sa_fake).to(self.device))
		dis_loss = self.criterion(dis_real, self.ones_label) + self.criterion(dis_fake, self.zeros_label)

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

		#Train discriminator
		self.opt_dis.zero_grad()
		dis_loss.backward()
		self.opt_dis.step()

		return pg_loss.item(), v_loss.item(), ac_loss.item(), ent.item(), dis_loss.item()


	#-----------------------
	# Learning rate decay
	#-----------------------
	def lr_decay(self, it, n_it):
		linear_lr_decay(self.opt_actor, it, n_it, self.lr)
		linear_lr_decay(self.opt_critic, it, n_it, self.lr)
		linear_lr_decay(self.opt_dis, it, n_it, self.lr)


#PPO Agent Class
class PPO:
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(
		self, 
		policy_net, 
		value_net, 
		dis_net,
		device, 
		a_dim,
		lr=1e-4, 
		max_grad_norm=0.5, 
		ent_weight=0.01,
		clip_val=0.2,
		sample_n_epoch=4,
		sample_mb_size=64,
		mb_size=1024,
		conti=False
	):
		self.opt_actor      = torch.optim.Adam(policy_net.parameters(), lr)
		self.opt_critic     = torch.optim.Adam(value_net.parameters(), lr)
		self.opt_dis        = torch.optim.Adam(dis_net.parameters(), lr)
		self.device         = device
		self.a_dim          = a_dim
		self.lr             = lr
		self.max_grad_norm  = max_grad_norm
		self.ent_weight     = ent_weight
		self.clip_val       = clip_val
		self.sample_n_epoch = sample_n_epoch
		self.sample_mb_size = sample_mb_size
		self.sample_n_mb    = mb_size // sample_mb_size
		self.rand_idx       = np.arange(mb_size)
		self.criterion      = nn.BCELoss()
		self.ones_label     = torch.autograd.Variable(torch.ones((sample_mb_size, 1))).to(self.device)
		self.zeros_label    = torch.autograd.Variable(torch.zeros((sample_mb_size, 1))).to(self.device)
		self.conti          = conti


	#-----------------------
	# Train PPO
	#-----------------------
	def train(
		self, 
		policy_net, 
		value_net, 
		dis_net,
		mb_obs, 
		mb_actions, 
		mb_old_values, 
		mb_advs, 
		mb_returns, 
		mb_old_a_logps, 
		sa_real
	):
		mb_obs         = torch.from_numpy(mb_obs).to(self.device)
		mb_actions     = torch.from_numpy(mb_actions).to(self.device)
		mb_old_values  = torch.from_numpy(mb_old_values).to(self.device)
		mb_advs        = torch.from_numpy(mb_advs).to(self.device)
		mb_returns     = torch.from_numpy(mb_returns).to(self.device)
		mb_old_a_logps = torch.from_numpy(mb_old_a_logps).to(self.device)

		#Train PPO
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

				ac_loss = pg_loss + 0.2*v_loss

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

		#Train Discriminator
		np.random.shuffle(self.rand_idx)

		for i in range(self.sample_n_mb):
			sample_idx     = self.rand_idx[i*self.sample_mb_size : (i+1)*self.sample_mb_size]
			sample_obs     = mb_obs[sample_idx]
			sample_actions = mb_actions[sample_idx]

			if self.conti:
				mb_sa_fake = torch.cat([sample_obs, sample_actions], 1)
			else:
				sample_actions_onehot = np.zeros([self.sample_mb_size, self.a_dim], dtype=np.float32)

				for j in range(self.sample_mb_size):
					sample_actions_onehot[j, sample_actions[j]] = 1

				mb_sa_fake = torch.cat([sample_obs, torch.from_numpy(sample_actions_onehot).to(self.device)], 1)

			mb_sa_real = sa_real[np.random.randint(0, sa_real.shape[0], self.sample_mb_size), :]

			#Adversarial loss
			dis_real = dis_net(torch.from_numpy(mb_sa_real).to(self.device))
			dis_fake = dis_net(mb_sa_fake)
			dis_loss = self.criterion(dis_real, self.ones_label) + self.criterion(dis_fake, self.zeros_label)

			self.opt_dis.zero_grad()
			dis_loss.backward()
			self.opt_dis.step()

		return pg_loss.item(), v_loss.item(), ac_loss.item(), ent.item(), dis_loss.item(), dis_real.mean().item(), dis_fake.mean().item()


	#-----------------------
	# Learning rate decay
	#-----------------------
	def lr_decay(self, it, n_it):
		linear_lr_decay(self.opt_actor, it, n_it, self.lr)
		linear_lr_decay(self.opt_critic, it, n_it, self.lr)