from env_runner import EnvRunner
from model import PolicyNet, ValueNet
from agent import PPO
import torch
import os
import gym
import time
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
    clip_val       = 0.2
    sample_mb_size = 64
    sample_n_epoch = 4
    lamb           = 0.95
    gamma          = 0.99
    ent_weight     = 0.01
    max_grad_norm  = 0.5
    lr             = 1e-4
    n_iter         = 10000
    disp_step      = 30
    save_step      = 300
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

    runner = EnvRunner(s_dim, a_dim, gamma, lamb, max_step=2048, device=device, conti=not args.discrete)

    #Create model
    #----------------------------
    policy_net = PolicyNet(s_dim, a_dim, conti=not args.discrete).to(device)
    value_net  = ValueNet(s_dim).to(device)
    agent      = PPO(
        policy_net,
        value_net,
        lr,
        max_grad_norm,
        ent_weight,
        clip_val,
        sample_n_epoch,
        sample_mb_size,
        device=device
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
        start_it = checkpoint["it"]
        print("Done.")
    else:
        start_it = 0

    #Start training
    #----------------------------
    t_start = time.time()
    policy_net.train()
    value_net.train()
    mean_total_reward = 0
    mean_length = 0

    for it in range(start_it, n_iter):
        #Run the environment
        with torch.no_grad():
            mb_obs, mb_actions, mb_old_a_logps, mb_values, mb_returns, mb_rewards = runner.run(
                env, policy_net, value_net
            )
            mb_advs = mb_returns - mb_values
            mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-6)

        #Train
        pg_loss, v_loss, ent = agent.train(
            policy_net,
            value_net,
            mb_obs,
            mb_actions,
            mb_values,
            mb_advs,
            mb_returns,
            mb_old_a_logps
        )
        mean_total_reward += mb_rewards.sum()
        mean_length += len(mb_obs)
        print("[Episode {:4d}] total reward = {:.6f}, length = {:d}".format(it, mb_rewards.sum(), len(mb_obs)))

        #Print the result
        if it % disp_step == 0:
            print("\n[{:5d} / {:5d}]".format(it, n_iter))
            print("----------------------------------")
            print("Elapsed time = {:.2f} sec".format(time.time() - t_start))
            print("actor loss   = {:.6f}".format(pg_loss))
            print("critic loss  = {:.6f}".format(v_loss))
            print("entropy      = {:.6f}".format(ent))
            print("mean return  = {:.6f}".format(mean_total_reward / disp_step))
            print("mean length  = {:.2f}".format(mean_length / disp_step))
            print()

            agent.lr_decay(it, n_iter)
            mean_total_reward = 0
            mean_length = 0

        #Save model
        if it % save_step == 0:
            print("Saving the model ... ", end="")
            torch.save({
                "it": it,
                "PolicyNet": policy_net.state_dict(),
                "ValueNet": value_net.state_dict()
            }, os.path.join(save_dir, "{}.pt".format(args.env)))
            print("Done.")
            print()

    env.close()

if __name__ == '__main__':
    main()
