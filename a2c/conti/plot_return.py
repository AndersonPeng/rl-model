from matplotlib import pyplot as plt
import os
import argparse


#Parse arguments
#----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="BipedalWalker-v2")
args = parser.parse_args()


#Read file
#----------------------------
env_id = args.env
save_dir = "./save_" + env_id
return_fp = return_fp = open(os.path.join(save_dir, "avg_return.txt"), "r")


#Plot
#----------------------------
avg_return = [float(t) for t in return_fp.read().split("\n")[:-1]]

running_return = [avg_return[0]]
for i in range(1, len(avg_return)):
	running_return.append(0.95*running_return[-1] + 0.05*avg_return[i])

plt.plot(avg_return)
plt.plot(running_return)
plt.show()