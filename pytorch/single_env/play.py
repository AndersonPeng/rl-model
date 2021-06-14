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
    parser.add_argument("--env", default="BipedalWalker-v3")
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--unwrap", action="store_true")
    args = parser.parse_args()

    #Parameters
    #----------------------------
    save_dir       = "./save"
    device         = "cuda:0" if torch.cuda.is_available() else "cpu"

    #Create environment
    #----------------------------
    env = gym.make(args.env)

    if args.discrete:
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.n
    else:
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]

    if args.unwrap:
        env = env.unwrapped

    #Create model
    #----------------------------
    policy_net = PolicyNet(s_dim, a_dim, conti=not args.discrete).to(device)
    print(policy_net)

    #Load model
    #----------------------------
    model_path = os.path.join(save_dir, "{}.pt".format(args.env))

    if os.path.exists(model_path):
        print("Loading the model ... ", end="")
        checkpoint = torch.load(model_path)
        policy_net.load_state_dict(checkpoint["PolicyNet"])
        start_it = checkpoint["it"]
        print("Done.")
    else:
        print("Error: No model saved")
        os.exit(1)

    #Start training
    #----------------------------
    policy_net.eval()

    with torch.no_grad():
        for it in range(10):
            ob = env.reset()
            total_reward = 0
            length = 0

            while True:
                env.render()
                ob_tensor = torch.tensor(np.expand_dims(ob, axis=0), dtype=torch.float32, device=device)
                action = policy_net.action_step(ob_tensor, deterministic=True).cpu().numpy()
                ob, reward, done, info = env.step(action[0])
                total_reward += reward
                length += 1

                if done:
                    print("Total reward = {:.6f}, length = {:d}".format(total_reward, length))
                    break

    env.close()

if __name__ == '__main__':
    main()
