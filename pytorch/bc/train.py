from model import PolicyNet
import torch
import os
import sys
import gym
import time
import argparse
import numpy as np
import pickle as pkl


def sample_batch(s_traj, a_traj, mb_size=256):
	rand_idx = np.arange(len(s_traj))
	np.random.shuffle(rand_idx)

	return s_traj[rand_idx[:mb_size]], a_traj[rand_idx[:mb_size]]


#-----------------------
# Main function
#-----------------------
def main():
	#Parse arguments
	#----------------------------
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="CartPole-v0")
	parser.add_argument("--conti", action="store_true")
	args = parser.parse_args()

	#Parameters
	#----------------------------
	env_id      = args.env
	mb_size     = 256
	lr          = 1e-5
	n_iter      = 100000
	disp_step   = 1000
	save_step   = 10000
	save_dir    = "./save"
	device      = "cuda:0"
	expert_path = "../save/{}_traj.pkl".format(args.env)

	#Create environment
	#----------------------------
	env = gym.make(env_id)
	
	if args.conti:
		s_dim = env.observation_space.shape[0]
		a_dim = env.action_space.shape[0]
	else:
		s_dim = env.observation_space.shape[0]
		a_dim = env.action_space.n

	#Load expert trajectories
	#----------------------------
	if os.path.exists(expert_path):
		s_traj, a_traj = pkl.load(open(expert_path, "rb"))
		s_traj = np.concatenate(s_traj, 0)
		a_traj = np.concatenate(a_traj, 0)
	else:
		print("ERROR: No expert trajectory file found")
		sys.exit(1)

	#Create model
	#----------------------------
	policy_net = PolicyNet(s_dim, a_dim, conti=args.conti).to(device)
	opt        = torch.optim.Adam(policy_net.parameters(), lr)

	#Load model
	#----------------------------
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	if os.path.exists(os.path.join(save_dir, "{}.pt".format(args.env))):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(os.path.join(save_dir, "{}.pt".format(args.env)))
		policy_net.load_state_dict(checkpoint["PolicyNet"])
		start_it = checkpoint["it"]
		print("Done.")
	else:
		start_it = 0

	#Start training
	#----------------------------
	t_start = time.time()
	policy_net.train()

	for it in range(start_it, n_iter+1):
		#Train
		mb_obs, mb_actions  = sample_batch(s_traj, a_traj, mb_size)
		mb_a_logps, mb_ents = policy_net.evaluate(
			torch.from_numpy(mb_obs).to(device), 
			torch.from_numpy(mb_actions).to(device)
		)
		loss = -mb_a_logps.mean()

		opt.zero_grad()
		loss.backward()
		opt.step()

		#Print the result
		if it % disp_step == 0:
			print("[{:5d} / {:5d}] Elapsed time = {:.2f}, actor loss = {:.6f}".format(
				it, n_iter, time.time() - t_start, loss.item())
			)

		#Save model
		if it % save_step == 0:
			print("Saving the model ... ", end="")
			torch.save({
				"it": it,
				"PolicyNet": policy_net.state_dict()
			}, os.path.join(save_dir, "{}.pt".format(args.env)))
			print("Done.")
			print()

	env.close()


if __name__ == '__main__':
	main()