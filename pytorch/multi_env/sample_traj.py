from model import PolicyNet
import torch
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
	parser.add_argument("--episode", default=1000)
	args = parser.parse_args()

	#Parameters
	#----------------------------
	env_id    = args.env
	save_dir  = "./save"
	device    = "cuda:0"
	n_episode = args.episode

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
	policy_net = PolicyNet(s_dim, a_dim, conti=args.conti).to(device)

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
	s_traj = []
	a_traj = []

	for i_episode in range(n_episode):
		ob  = env.reset()
		ret = 0
		s_traj.append([])
		a_traj.append([])

		while True:
			if args.render:
				env.render()

			action = policy_net.action_step(torch.FloatTensor(np.expand_dims(ob, axis=0)).to(device), deterministic=True)
			action = action.cpu().detach().numpy()[0]

			s_traj[i_episode].append(ob)
			a_traj[i_episode].append(action)

			ob, reward, done, info = env.step(action)
			ret += reward

			if done:
				s_traj[i_episode] = np.array(s_traj[i_episode], dtype=np.float32)

				if args.conti:
					a_traj[i_episode] = np.array(a_traj[i_episode], dtype=np.float32)
				else:
					a_traj[i_episode] = np.array(a_traj[i_episode], dtype=np.int32)

				print("{:d}: return = {:.4f}, len = {:d}".format(i_episode, ret, len(s_traj[i_episode])))
				break

	#s_traj: (n_episode, timesteps, s_dim)
	#a_traj: (n_episode, timesteps, a_dim) or (n_episode, timesteps)
	print("Saving the trajectories ... ", end="")
	pkl.dump((s_traj, a_traj), open(os.path.join(save_dir, "{}_traj.pkl".format(env_id)), "wb"))
	print("Done.")
	env.close()

if __name__ == '__main__':
	main()
