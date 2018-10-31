import network

if __name__ =='__main__':
	train = network.Network(optimizer = 'GD')
	train.train([0.1, 0.2, 0.3, 1.87], [1, 0, 0, 0.5], epoch = 10000)
