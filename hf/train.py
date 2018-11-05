import random

# ======CALL BY network.py======
# 
# 
class Optimizer:
	def __init__(self, counter, batchSize, layers, use_func = "SGD", LR = 0.5):	
		index = {"SGD": self.stochasticGradientDescent, "miniBatchSGD": self.miniBatchStochasticGradientDescent}
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