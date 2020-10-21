from model import PolicyNet
import torch
import os
import gym
import argparse
import numpy as np


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
	env_id   = args.env
	save_dir = "./save"
	device   = "cuda:0"

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

	for it in range(100):
		ob  = env.reset()
		ret = 0

		while True:
			env.render()
			action = policy_net.action_step(torch.FloatTensor(np.expand_dims(ob, axis=0)).to(device), deterministic=True)
			ob, reward, done, info = env.step(action.cpu().detach().numpy()[0])
			ret += reward

			if done:
				print("return = {:.4f}".format(ret))
				break

	env.close()


if __name__ == '__main__':
	main()