import random
import hf.nn
class Saver:
	def __init__(self, network, savePath = './weight.txt'):
		file = open(savePath, 'w')
		for l in range(len(network)):
			for n in range(len(network[l].neurons)):
				file.write(str(network[l].neurons[n].weight) + '\n')
				file.write(str(network[l].neurons[n].bias) + '\n')
		file.close()

class SetBack:
	def __init__(self, network, filePath = './weight.txt'):
		file = open(filePath, 'r')
		for l in range(len(network)):
			for n in range(len(network[l].neurons)):
				weight = file.readline()[1:-2].split(',')
				weight = [float(w) for w in weight]
				network[l].neurons[n].weight = weight

				bias = float(file.readline()[:-1])
				network[l].neurons[n].bias = bias
		file.close()			

class BackPropagation:
	def __init__(self, network, loss, oneHotPos = None):
		self.reverseNet = network.copy()
		self.reverseNet.reverse()

		gradient = loss.backForward()
		
		for layer in self.reverseNet:
			if oneHotPos is None:
				gradient = layer.backPropagation(gradient)
			else:
				gradient = layer.backPropagation(gradient, oneHotPos)
				oneHotPos = None
				
class FeedForwardTraining:
	def __init__(self, network, input):
		layerOutput = network[0].feedForward(input)

		for i in range(1, len(network)):
			layerOutput = network[i].feedForward(layerOutput)

		self.predict = layerOutput
		
	def loss(self, target, loss = 'CE'):
		loss = hf.nn.Loss(self.predict, target, use_func = loss)
		return loss

	def getPredict(self):
		return self.predict

# ======CALL BY network.py======
# 
# 
class Optimizer:
	def __init__(self, counter, batchSize, layers, use_func = "SGD", LR = 0.5):	
		index = {"SGD": self.stochasticGradientDescent, "miniBatchSGD": self.miniBatchStochasticGradientDescent,

				}
		optimizer = index.get(use_func, "Optimizer does not exist.")
		self.counter = counter
		self.batchSize = batchSize
		self.layers = layers
		self.LR = LR
		optimizer()

	def stochasticGradientDescent(self):
		# choose = random.randrange(len(self.layers[0].neurons[0].gradient[0]))
		# for l in range(len(self.layers)):
		# 	for n in range(len(self.layers[l].neurons)):
		# 		self.layers[l].neurons[n].bias -= self.LR* self.layers[l].neurons[n].derivation[choose]
		# 		self.layers[l].neurons[n].derivation = []
		# 		for g in range(len(self.layers[l].neurons[n].gradient)):
		# 			self.layers[l].neurons[n].weight[g] -= self.LR* self.layers[l].neurons[n].gradient[g][choose]
		# 			self.layers[l].neurons[n].gradient[g] = []
		for l in range(len(self.layers)):
			for n in range(len(self.layers[l].neurons)):
				self.layers[l].neurons[n].bias -= self.LR* self.layers[l].neurons[n].derivation[0]
				self.layers[l].neurons[n].derivation = []
				for g in range(len(self.layers[l].neurons[n].gradient)):
					self.layers[l].neurons[n].weight[g] -= self.LR* self.layers[l].neurons[n].gradient[g][0]
					self.layers[l].neurons[n].gradient[g] = []

	def miniBatchStochasticGradientDescent(self):
		if self.counter == (self.batchSize - 1):
			for l in range(len(self.layers)):
				for n in range(len(self.layers[l].neurons)):
					self.layers[l].neurons[n].bias -= self.LR* sum(self.layers[l].neurons[n].derivation)/ len(self.layers[l].neurons[n].derivation)
					self.layers[l].neurons[n].derivation = []
					for g in range(len(self.layers[l].neurons[n].gradient)):
						self.layers[l].neurons[n].weight[g] -= self.LR* sum(self.layers[l].neurons[n].gradient[g])/ len(self.layers[l].neurons[n].gradient[g])
						self.layers[l].neurons[n].gradient[g] = []