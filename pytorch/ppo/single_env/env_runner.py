import torch
import gym
import numpy as np

#-----------------------
# Compute discounted return
#-----------------------
def compute_discounted_return(rewards, last_values, gamma=0.99):
    returns = np.zeros_like(rewards)
    n_step  = len(rewards)

    for t in reversed(range(n_step)):
        if t == n_step - 1:
            returns[t] = rewards[t] + gamma * last_values
        else:
            returns[t] = rewards[t] + gamma * returns[t+1]

    return returns

#-----------------------
# Compute gae
#-----------------------
def compute_gae(rewards, values, last_value, gamma=0.99, lamb=0.95):
    advs         = np.zeros_like(rewards)
    n_step       = len(rewards)
    last_gae_lam = 0.0

    for t in reversed(range(n_step)):
        if t == n_step - 1:
            next_value = last_value
        else:
            next_value = values[t+1]

        delta   = rewards[t] + gamma*next_value - values[t]
        advs[t] = last_gae_lam = delta + gamma*lamb*last_gae_lam

    return advs + values

#Runner for multiple environment
class EnvRunner:
    #-----------------------
    # Constructor
    #-----------------------
    def __init__(self, s_dim, a_dim, gamma=0.99, lamb=0.95, max_step=1024, device="cuda:0", conti=False):
        self.s_dim    = s_dim
        self.a_dim    = a_dim
        self.gamma    = gamma
        self.lamb     = lamb
        self.max_step = max_step
        self.device   = device
        self.conti    = conti

        #Storages (state, action, value, reward, a_logp)
        self.mb_obs     = np.zeros((self.max_step, self.s_dim), dtype=np.float32)
        self.mb_values  = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_rewards = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_a_logps = np.zeros((self.max_step,), dtype=np.float32)

        if conti:
            self.mb_actions = np.zeros((self.max_step, self.a_dim), dtype=np.float32)
        else:
            self.mb_actions = np.zeros((self.max_step,), dtype=np.int32)

    #-----------------------
    # Run n steps to get a batch
    #-----------------------
    def run(self, env, policy_net, value_net, render=False):
        #1. Run an episode
        #-------------------------------------
        ob = env.reset()   #Initial state
        episode_len = self.max_step

        for step in range(self.max_step):
            if render: env.render()

            ob_tensor = torch.tensor(np.expand_dims(ob, axis=0), dtype=torch.float32, device=self.device)
            action, a_logp = policy_net(ob_tensor)
            value = value_net(ob_tensor)

            action = action.cpu().numpy()[0]
            a_logp = a_logp.cpu().numpy()
            value  = value.cpu().numpy()

            self.mb_obs[step]     = ob
            self.mb_actions[step] = action
            self.mb_a_logps[step] = a_logp
            self.mb_values[step]  = value

            ob, reward, done, info = env.step(action)
            self.mb_rewards[step] = reward

            if done:
                episode_len = step + 1
                break

        #2. Compute returns
        #-------------------------------------
        last_value = value_net(
            torch.tensor(np.expand_dims(ob, axis=0), dtype=torch.float32, device=self.device)
        ).cpu().numpy()

        mb_returns = compute_gae(
            self.mb_rewards[:episode_len],
            self.mb_values[:episode_len],
            last_value,
            self.gamma,
            self.lamb
        )

        return self.mb_obs[:episode_len], \
                self.mb_actions[:episode_len], \
                self.mb_a_logps[:episode_len], \
                self.mb_values[:episode_len], \
                mb_returns, \
                self.mb_rewards[:episode_len]
