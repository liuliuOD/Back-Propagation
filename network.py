import hf.layer
import hf.nn
import hf.train
import math

class Network:
	def __init__(self, optimizer = "SGD"):
		self.h1 = hf.layer.Layer(4, 4, active = "relu")
		# self.h2 = hf.layer.Layer(2, 2, active = "sigmoid")
		# self.h3 = hf.layer.Layer(4, 4, active = "sigmoid")
		self.o1 = hf.layer.Layer(4, 4, active = "softmax")
		self.optimizer = optimizer

		# self.layers = [self.h1, self.h2, self.h3, self.o1]
		self.layers = [self.h1, self.o1]
	def train(self, input, target, batch = 1):
		meanLoss = 0
		predictTrue = 0
		Otn = []

		for numBatch in range(math.ceil(len(input)/ batch)):
			for i in range(batch):
				h1 = self.h1.feedForward(input[numBatch* batch + i])
				# h2 = self.h2.feedForward(h1)
				# h3 = self.h3.feedForward(h2)
				# o1 = self.o1.feedForward(h3)
				o1 = self.o1.feedForward(h1)
				loss = hf.nn.Loss(o1, target[numBatch* batch + i], use_func = 'CE')

				gradient = loss.backForward()
				# print("O1")
				gradient = self.o1.backPropagation(gradient, target[numBatch* batch + i].index(max(target[numBatch* batch + i])))
				# print("H3")
				# gradient = self.h3.backPropagation(gradient)
				# # print("H2")
				# gradient = self.h2.backPropagation(gradient)
				# print("H1")
				gradient = self.h1.backPropagation(gradient)

				meanLoss += loss.get()
				Otn.append(o1)
				hf.train.Optimizer(i, batch, self.layers, use_func = self.optimizer, LR =0.5)

		return meanLoss/ batch, Otn
		

if __name__ == '__main__':
	train = Network()
	train.train([0.1, 0.2, 0.3], [1, 0, 0], epoch = 10000)