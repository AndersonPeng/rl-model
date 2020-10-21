from multi_env import MultiEnv, make_env
from env_runner import EnvRunner
from model import PolicyNet, ValueNet, DiscriminatorNet
from agent import PPO
import torch
import os
import sys
import gym
import time
import argparse
import numpy as np
import pickle as pkl


#-----------------------
# Main function
#-----------------------
def main():
	#Parse arguments
	#----------------------------
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="CartPole-v0")
	parser.add_argument("--conti", action="store_true")
	parser.add_argument("--unwrap", action="store_true")
	args = parser.parse_args()

	#Parameters
	#----------------------------
	n_env          = 8
	n_step         = 128
	mb_size        = n_env*n_step
	sample_mb_size = 64
	sample_n_epoch = 4
	clip_val       = 0.2
	lamb           = 0.95
	gamma          = 0.99
	ent_weight     = 0.0
	max_grad_norm  = 0.5
	lr             = 1e-4
	n_iter         = 30000
	disp_step      = 30
	save_step      = 300
	save_dir       = "./save"
	device         = "cuda:0"
	expert_path    = "../save/{}_traj.pkl".format(args.env)

	#Create multiple environments
	#----------------------------
	env = MultiEnv([make_env(i, env_id=args.env, unwrap=args.unwrap, rand_seed=int(time.time())) for i in range(n_env)])
	
	if args.conti:
		s_dim = env.ob_space.shape[0]
		a_dim = env.ac_space.shape[0]
	else:
		s_dim = env.ob_space.shape[0]
		a_dim = env.ac_space.n

	runner = EnvRunner(
		env, 
		s_dim, 
		a_dim,
		n_step, 
		gamma,
		lamb,
		device=device, 
		conti=args.conti
	)

	#Load expert trajectories
	#----------------------------
	if os.path.exists(expert_path):
		s_real, a_real = pkl.load(open(expert_path, "rb"))
		sa_real = []

		if args.conti:
			for i in range(len(s_real)):
				sa_real.append(np.concatenate([s_real[i], a_real[i]], 1))
		else:
			for i in range(len(s_real)):
				a_real_onehot = np.zeros((len(a_real[i]), a_dim), dtype=np.float32)

				for j in range(len(a_real[i])):
					a_real_onehot[j, a_real[i][j]] = 1

				sa_real.append(np.concatenate([s_real[i], a_real_onehot], 1))

		sa_real = np.concatenate(sa_real, 0)
	else:
		print("ERROR: No expert trajectory file found")
		sys.exit(1)

	#Create model
	#----------------------------
	policy_net = PolicyNet(s_dim, a_dim, conti=args.conti).to(device)
	value_net  = ValueNet(s_dim).to(device)
	dis_net    = DiscriminatorNet(s_dim+a_dim).to(device)
	agent      = PPO(
		policy_net, 
		value_net,
		dis_net, 
		a_dim, 
		lr, 
		max_grad_norm, 
		ent_weight, 
		clip_val, 
		sample_n_epoch, 
		sample_mb_size, 
		mb_size,
		device=device,
		conti=args.conti
	)

	#Load model
	#----------------------------
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	if os.path.exists(os.path.join(save_dir, "{}.pt".format(args.env))):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(os.path.join(save_dir, "{}.pt".format(args.env)))
		policy_net.load_state_dict(checkpoint["PolicyNet"])
		value_net.load_state_dict(checkpoint["ValueNet"])
		dis_net.load_state_dict(checkpoint["DiscriminatorNet"])
		start_it = checkpoint["it"]
		print("Done.")
	else:
		start_it = 0

	#Start training
	#----------------------------
	t_start = time.time()
	policy_net.train()
	value_net.train()

	for it in range(start_it, n_iter):
		#Run the environment
		with torch.no_grad():
			mb_obs, mb_actions, mb_old_a_logps, mb_values, mb_returns = runner.run(policy_net, value_net, dis_net)
			mb_advs = mb_returns - mb_values
			mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-6)

		#Train
		pg_loss, v_loss, ent, dis_loss, dis_real, dis_fake = agent.train(
			policy_net, 
			value_net, 
			dis_net,
			mb_obs, 
			mb_actions, 
			mb_values,
			mb_advs, 
			mb_returns,
			mb_old_a_logps,
			sa_real
		)

		#Print the result
		if it % disp_step == 0:
			agent.lr_decay(it, n_iter)
			policy_net.eval()
			value_net.eval()
			n_sec = time.time() - t_start
			fps = int((it - start_it)*n_env*n_step / n_sec)
			mean_true_return, std_true_return, mean_return, std_return, mean_len = runner.get_performance()
			policy_net.train()
			value_net.train()

			print("[{:5d} / {:5d}]".format(it, n_iter))
			print("----------------------------------")
			print("Timesteps        = {:d}".format((it - start_it) * mb_size))
			print("Elapsed time     = {:.2f} sec".format(n_sec))
			print("FPS              = {:d}".format(fps))
			print("actor loss       = {:.6f}".format(pg_loss))
			print("critic loss      = {:.6f}".format(v_loss))
			print("dis loss         = {:.6f}".format(dis_loss))
			print("entropy          = {:.6f}".format(ent))
			print("mean true return = {:.6f}".format(mean_true_return))
			print("mean return      = {:.6f}".format(mean_return))
			print("mean length      = {:.2f}".format(mean_len))
			print("dis_real         = {:.3f}".format(dis_real))
			print("dis_fake         = {:.3f}".format(dis_fake))
			print()

		#Save model
		if it % save_step == 0:
			print("Saving the model ... ", end="")
			torch.save({
				"it": it,
				"PolicyNet": policy_net.state_dict(),
				"ValueNet": value_net.state_dict(),
				"DiscriminatorNet": dis_net.state_dict()
			}, os.path.join(save_dir, "{}.pt".format(args.env)))
			print("Done.")
			print()

	env.close()


if __name__ == '__main__':
	main()