from matplotlib import pyplot as plt
import numpy as np
import os
import sys


#Read file
#----------------------------
if len(sys.argv) < 2:
	print("ERROR: No input path")
	sys.exit(1)

save_dir  = sys.argv[1]
return_fp = return_fp = open(os.path.join(save_dir, "avg_return.txt"), "r")

avg_return = []
std_return = []

for line in return_fp.read().split("\n")[:-1]:
	vals = line.split(",")
	avg_return.append(float(vals[0]))
	std_return.append(float(vals[1]))

running_return = [avg_return[0]]
for i in range(1, len(avg_return)):
	running_return.append(0.95*running_return[-1] + 0.05*avg_return[i])

avg_return = np.array(avg_return)
std_return = np.array(std_return)
return_fp.close()


#Plot
#----------------------------
p1, = plt.plot(avg_return, color=(0.8, 0.0, 0.0, 0.5))
p2, = plt.plot(running_return, color=(0.0, 0.0, 0.5, 1.0))
plt.fill_between(range(len(avg_return)), avg_return-std_return, avg_return+std_return, color=(0.8, 0.0, 0.0, 0.2))
plt.legend([p1, p2], ["avg. return", "running avg. return"])
plt.grid()
plt.show()