# rl-model

The implementation is based on [OpenAI Baselines](https://github.com/openai/baselines)

<br>

## Prerequisite

- tensorflow-1.8.0
- gym-0.10.5
- mujoco-py-1.50.1 (optional)

<br>

## Models

- Advantage Actor-Critic (A2C) (Original paper: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783))
- Proximal Policy Optimization (PPO) (Original paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347))

<br>

## Experiment Results

### Discrete:

- CartPole-v0

A2C|PPO
-|-
![](./fig/CartPole-v0-a2c.png)|![](./fig/CartPole-v0-ppo.png)

<br>

- MountainCar-v0

A2C|PPO
-|-
![](./fig/MountainCar-v0-a2c.png)|![](./fig/MountainCar-v0-ppo.png)


<br>

- Acrobot-v1

A2C|PPO
-|-
![](./fig/Acrobot-v1-a2c.png)|![](./fig/Acrobot-v1-ppo.png)

<br>
<br>

### Atari (Discrete):

- BreakoutNoFrameskip-v4

A2C|PPO
-|-
![](./fig/BreakoutNoFrameskip-v4-a2c.png)|![](./fig/BreakoutNoFrameskip-v4-ppo.png)

<br>

- PongNoFrameskip-v4

A2C|PPO
-|-
![](./fig/PongNoFrameskip-v4-a2c.png)|![](./fig/PongNoFrameskip-v4-ppo.png)

<br>

- SpaceInvadersNoFrameskip-v4

A2C|PPO
-|-
![](./fig/SpaceInvadersNoFrameskip-v4-a2c.png)|![](./fig/SpaceInvadersNoFrameskip-v4-ppo.png)

<br>
<br>

### Continuous:

- InvertedPendulum-v2

A2C|PPO
-|-
![](./fig/InvertedPendulum-v2-a2c.png)|![](./fig/InvertedPendulum-v2-ppo.png)

<br>

- InvertedDoublePendulum-v2

A2C|PPO
-|-
![](./fig/InvertedDoublePendulum-v2-a2c.png)|![](./fig/InvertedDoublePendulum-v2-ppo.png)

<br>

- BipedalWalker-v2

A2C|PPO
-|-
![](./fig/BipedalWalker-v2-a2c.png)|![](./fig/BipedalWalker-v2-ppo.png)

<br>

- HalfCheetah-v2

A2C|PPO
-|-
![](./fig/HalfCheetah-v2-a2c.png)|![](./fig/HalfCheetah-v2-ppo.png)

<br>

- Hopper-v2

A2C|PPO
-|-
![](./fig/Hopper-v2-a2c.png)|![](./fig/Hopper-v2-ppo.png)
