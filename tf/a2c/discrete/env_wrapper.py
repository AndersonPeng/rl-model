import numpy as np
import gym
import cv2
from collections import deque
from gym import spaces


#Sample initial states by taking random number of no-ops on reset
class NoopResetEnv(gym.Wrapper):
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(self, env, noop_max=30):
		gym.Wrapper.__init__(self, env)
		self.noop_max = noop_max
		self.override_n_noop = None
		self.noop_action = 0

		#no-op is assumed to be action 0
		#Action 0: NOOP
		assert(env.unwrapped.get_action_meanings()[0] == "NOOP")


	#--------------------------
	# Reset
	# Do no-op action for a number
	# of steps in [1, n_noop]
	#--------------------------
	def _reset(self, **kwargs):
		self.env.reset(**kwargs)

		if self.override_n_noop is not None:
			n_noop = self.override_n_noop
		else:
			n_noop = self.unwrapped.np_random.randint(1, self.noop_max + 1)

		assert(n_noop > 0)

		#Do NOOP n_noop times
		obs = None
		for i in range(n_noop):
			obs, _, done, _ = self.env.step(self.noop_action)
			if done:
				obs = self.env.reset(**kwargs)

		return obs


#Take action on reset for environments that are fixed until firing
class FireResetEnv(gym.Wrapper):
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)
		assert(env.unwrapped.get_action_meanings()[1] == "FIRE")
		assert(len(env.unwrapped.get_action_meanings()) >= 3)


	#--------------------------
	# Reset
	#--------------------------
	def _reset(self, **kwargs):
		self.env.reset(**kwargs)

		#Action 1: FIRE
		obs, _, done, _ = self.env.step(1)
		if done:
			self.env.reset(**kwargs)

		#Action 2: UP
		obs, _, done, _ = self.env.step(2)
		if done:
			self.env.reset(**kwargs)

		return obs


#Make end-of-life == end-of-episode, but only reset on true game over
class EpisodicLifeEnv(gym.Wrapper):
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)
		self.lives = 0
		self.was_real_done = True


	#--------------------------
	# Step
	#--------------------------
	def _step(self, action):
		obs, reward, done, info = self.env.step(action)
		self.was_real_done = done

		#Check current lives, make loss of life terminal
		#Then update lives to handle bonus lives
		lives = self.env.unwrapped.ale.lives()
		if lives < self.lives and lives > 0:
			done = True

		self.lives = lives
		return obs, reward, done, info


	#--------------------------
	# Reset
	#--------------------------
	def _reset(self, **kwargs):
		if self.was_real_done:
			obs = self.env.reset(**kwargs)
		else:
			#Action 0: NOOP
			#no-op step to advance from terminal/lost life state
			obs, _, _, _ = self.env.step(0)

		self.lives = self.env.unwrapped.ale.lives()
		return obs


#Return only every 'skip'-th frame
class MaxAndSkipEnv(gym.Wrapper):
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(self, env, skip=4):
		gym.Wrapper.__init__(self, env)
		self._obs_buf = np.zeros((2,) + env.observation_space.shape, dtype="uint8")
		self._skip = skip


	#--------------------------
	# Step
	#--------------------------
	def _step(self, action):
		total_reward = 0.0
		done = None

		for i in range(self._skip):
			obs, reward, done, info = self.env.step(action)

			if i == self._skip - 2:
				self._obs_buf[0] = obs

			if i == self._skip - 1:
				self._obs_buf[1] = obs

			total_reward += reward

			if done:
				break

		#Max over last observations
		max_frame = self._obs_buf.max(axis=0)

		return max_frame, total_reward, done, info


#Stack last k frames, return lazy array
class FrameStack(gym.Wrapper):
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(self, env, k):
		gym.Wrapper.__init__(self, env)
		self.k = k
		self.frames = deque([], maxlen=k)

		obs_shape = env.observation_space.shape
		self.observation_space = spaces.Box(low=0, high=255, shape=(obs_shape[0], obs_shape[1], obs_shape[2]*k))


	#--------------------------
	# Reset
	#--------------------------
	def _reset(self):
		obs = self.env.reset()

		#Enqueue k frames
		for _ in range(self.k):
			self.frames.append(obs)

		return self._get_ob()


	#--------------------------
	# Step
	#--------------------------
	def _step(self, action):
		obs, reward, done, info = self.env.step(action)

		#Enqueue a frame
		self.frames.append(obs)

		return self._get_ob(), reward, done, info


	#--------------------------
	# Get k-stacked observation
	#--------------------------
	def _get_ob(self):
		assert(len(self.frames) == self.k)
		return LazyFrames(list(self.frames))


#Bin reward to {+1, 0, -1}
class ClipRewardEnv(gym.RewardWrapper):
	#--------------------------
	# Reward
	#--------------------------
	def _reward(self, reward):
		return np.sign(reward)


#Warp frames to 84x84
class WarpFrame(gym.ObservationWrapper):
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(self, env):
		gym.ObservationWrapper.__init__(self, env)
		self.width = 84
		self.height = 84
		self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)


	#--------------------------
	# Observation
	#--------------------------
	def _observation(self, frame):
		#frame: (width, height, 1)
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
		return frame[:, :, None]


#Ensure conmmon frames between the observations are only stored once
class LazyFrames():
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(self, frames):
		self._frames = frames


	#--------------------------
	# Array
	#--------------------------
	def __array__(self, dtype=None):
		out = np.concatenate(self._frames, axis=2)

		if dtype is not None:
			out = out.astype(dtype)
		return out