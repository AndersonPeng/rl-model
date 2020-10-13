from model import PolicyNet
import torch
import torch.nn as nn
import numpy as np
import os
import gym
import argparse
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
	parser.add_argument("--render", action="store_true")
	parser.add_argument("--unwrap", action="store_true")
	args = parser.parse_args()

	#Parameters
	#----------------------------
	env_id    = args.env
	save_dir  = "./save"
	device    = "cuda:0"
	n_episode = 1000

	#Create environment
	#----------------------------
	env = gym.make(env_id)
	
	if args.conti:
		s_dim = env.observation_space.shape[0]
		a_dim = env.action_space.shape[0]
	else:
		s_dim = env.observation_space.shape[0]
		a_dim = env.action_space.n
	
	if args.unwrap:
		env = env.unwrapped

	#Create model
	#----------------------------
	policy_net = PolicyNet(s_dim, a_dim, args.conti).to(device)

	#Load model
	#----------------------------
	if os.path.exists(os.path.join(save_dir, "{}.pt".format(env_id))):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(os.path.join(save_dir, "{}.pt".format(env_id)))
		policy_net.load_state_dict(checkpoint["PolicyNet"])
		print("Done.")
	else:
		print("Error: No model saved")

	#Start playing
	#----------------------------
	policy_net.eval()
	sa_traj = []

	for it in range(n_episode):
		ob  = env.reset()
		ret = 0
		sa_traj.append([])

		while True:
			if args.render:
				env.render()

			action = policy_net.action_step(torch.from_numpy(np.expand_dims(ob.__array__(), axis=0)).float().to(device), deterministic=True)
			action = action.cpu().detach().numpy()[0]

			if args.conti:
				sa_traj[it].append(np.hstack([ob, action]))
			else:
				action_onehot = np.zeros([a_dim])
				action_onehot[action] = 1
				sa_traj[it].append(np.hstack([ob, action_onehot]))

			ob, reward, done, info = env.step(action)
			ret += reward

			if done:
				sa_traj[it] = np.array(sa_traj[it], dtype=np.float32)
				print("{:d}: return = {:.4f}, len = {:d}".format(it, ret, len(sa_traj[it])))
				break

	print("Saving the trajectories ... ", end="")
	pkl.dump(sa_traj, open(os.path.join(save_dir, "{}_traj.pkl".format(env_id)), "wb"))
	print("Done.")
	env.close()


if __name__ == '__main__':
	main()