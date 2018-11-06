import hf.layer
import hf.nn
import hf.train
import math

class Network:
	def __init__(self, optimizer = "SGD"):
		self.h1 = hf.layer.Layer(4, 1, active = "relu")
		self.h2 = hf.layer.Layer(1, 1, active = "relu")
		# self.h3 = hf.layer.Layer(1, 1, active = "relu")
		self.o1 = hf.layer.Layer(1, 4, active = "softmax")
		self.optimizer = optimizer

		self.layers = [self.h1, self.h2, self.o1]
		# self.layers = [self.h1, self.o1]
	def train(self, input, target, batch = 1):
		meanLoss = 0
		predictTrue = 0
		Otn = []

		for numBatch in range(math.ceil(len(input)/ batch)):
			for i in range(batch):
				dataNum = numBatch* batch + i
				oneHotPos = target[dataNum].index(max(target[dataNum]))

				# h1 = self.h1.feedForward(input[dataNum])
				# h2 = self.h2.feedForward(h1)
				# o1 = self.o1.feedForward(h2)
				# loss = hf.nn.Loss(o1, target[dataNum], use_func = 'CE')

				# gradient = loss.backForward()
				# gradient = self.o1.backPropagation(gradient, target[dataNum].index(max(target[dataNum])))
				# gradient = self.h1.backPropagation(gradient)
				
				train = hf.train.FeedForwardTraining(self.layers, input[dataNum])
				loss = train.loss(target[dataNum], loss = 'CE')

				hf.train.BackPropagation(self.layers, loss, oneHotPos)

				hf.train.Optimizer(i, batch, self.layers, use_func = self.optimizer, LR =0.5)

				meanLoss += loss.get()
				Otn.append(train.getPredict())

				# Otn.append(o1)
				

		return meanLoss/ batch, Otn
		

if __name__ == '__main__':
	train = Network()
	train.train([0.1, 0.2, 0.3], [1, 0, 0], epoch = 10000)