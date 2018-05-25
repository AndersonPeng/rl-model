import numpy as np
from multiprocessing import Process, Pipe


#---------------------------
# Worker
#---------------------------
def worker(remote, parent_remote, env_fn_wrapper):
	parent_remote.close()
	env = env_fn_wrapper.x()

	while True:
		cmd, data = remote.recv()

		if cmd == "step":
			ob, reward, done, info = env.step(data)
			if done:
				ob = env.reset()

			remote.send((ob, reward, done, info))

		elif cmd == "reset":
			ob = env.reset()
			remote.send(ob)

		elif cmd == "reset_task":
			ob = env.reset_task()
			remote.send(ob)

		elif cmd == "close":
			remote.close()
			break

		elif cmd == "get_spaces":
			remote.send((env.action_space, env.observation_space))

		elif cmd == "get_id":
			remote.send(env.spec.id)

		else:
			raise NotImplementedError


#To serialize contents (otherwise multiprocessing tries to use pickle)
class CloudpickleWrapper():
	def __init__(self, x):
		self.x = x

	def __getstate__(self):
		import cloudpickle
		return cloudpickle.dumps(self.x)

	def __setstate__(self, ob):
		import pickle
		self.x = pickle.loads(ob)


#List of gym envs to run in subprocesses
class MultiEnv():
	#---------------------------
	# Constructor
	#---------------------------
	def __init__(self, env_fns):
		#Create subprocesses
		self.closed = False
		self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(len(env_fns))])
		self.subprocs = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
						for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]

		#Start subprocesses
		for p in self.subprocs:
			p.deamon = True
			p.start()

		for remote in self.work_remotes:
			remote.close()

		#Get spaces & id
		self.remotes[0].send(("get_spaces", None))
		self.ac_space, self.ob_space = self.remotes[0].recv()

		self.remotes[0].send(("get_id", None))
		self.env_id = self.remotes[0].recv()


	#---------------------------
	# Step
	#---------------------------
	def step(self, actions):
		for remote, action in zip(self.remotes, actions):
			remote.send(("step", action))

		results = [remote.recv() for remote in self.remotes]
		obs, rewards, dones, infos = zip(*results)

		return np.stack(obs), np.stack(rewards), np.stack(dones), infos


	#---------------------------
	# Reset
	#---------------------------
	def reset(self):
		for remote in self.remotes:
			remote.send(("reset", None))

		return np.stack([remote.recv() for remote in self.remotes])


	#---------------------------
	# Reset task
	#---------------------------
	def reset_task(self):
		for remote in self.remotes:
			remote.send(("reset_task", None))

		return np.stack([remote.recv() for remote in self.remotes])


	#---------------------------
	# Close
	#---------------------------
	def close(self):
		if self.closed:
			return

		for remote in self.remotes:
			remote.send(("close", None))

		for p in self.subprocs:
			p.join()

		self.closed = True


	@property
	def n_env(self):
		return len(self.remotes)