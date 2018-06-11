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

![](./fig/CartPole-v0.png)

<br>

- MountainCar-v0

![](./fig/MountainCar-v0.png)

<br>
<br>

### Atari (Discrete):

- BreakoutNoFrameskip-v4

![](./fig/BreakoutNoFrameskip-v4.png)

<br>

- PongNoFrameskip-v4

![](./fig/PongNoFrameskip-v4.png)

<br>

- SpaceInvadersNoFrameskip-v4

![](./fig/SpaceInvadersNoFrameskip-v4.png)

<br>
<br>

### Continuous:

- InvertedPendulum-v2

![](./fig/InvertedPendulum-v2.png)

<br>

- BipedalWalker-v2

![](./fig/BipedalWalker-v2.png)