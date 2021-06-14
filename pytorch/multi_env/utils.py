#-----------------------
# Weight initialization
#-----------------------
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)

    return module

#-----------------------
# Linear learning rate decay
#-----------------------
def linear_lr_decay(opt, it, n_it, initial_lr):
	lr = initial_lr - (initial_lr * (it / float(n_it)))

	for param_group in opt.param_groups:
		param_group['lr'] = lr
