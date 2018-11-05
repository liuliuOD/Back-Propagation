import network

if __name__ =='__main__':
	train = network.Network(optimizer = 'miniBatchSGD')
	epoch =2000
	input = [[0.1, 0.9, 0.9, 0.89], [0.5, 0.6, 0.4, 0.13], [1.5, 1.3, 0.8, 0.5]]
	target = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]]
	# input = [[0.1, 0.2, 0.3, 0.89]]
	# target = [[1, 0, 0, 0]]

	for i in range(epoch):
		loss, output = train.train(input, target, batch = len(input))
		print("LOSS [{}]: {}".format(i, loss))
		print("OUTPUT [{}]: {}".format(i, output))
