import random

# ======CALL BY network.py======
# 
# 
class Optimizer:
	def __init__(self, layers, use_func = "SGD", LR =0.5):	
		index = {"SGD": self.stochasticGradientDescent, "miniBatchGD": self.miniBatchGradientDescent}
		optimizer = index.get(use_func, "Optimizer does not exist.")
		self.layers = layers
		self.LR = LR
		optimizer()

	def stochasticGradientDescent(self):
		choose = random.randrange(len(self.layers[0].neurons[0].gradient[0]))
		for l in range(len(self.layers)):
			for n in range(len(self.layers[l].neurons)):
				for g in range(len(self.layers[l].neurons[n].gradient)):
					self.layers[l].neurons[n].weight[g] -= self.LR* self.layers[l].neurons[n].gradient[g][choose]
					self.layers[l].neurons[n].gradient[g] = []

	def miniBatchGradientDescent(self):
		for l in range(len(self.layers)):
			for n in range(len(self.layers[l].neurons)):
				for g in range(len(self.layers[l].neurons[n].gradient)):
					self.layers[l].neurons[n].weight[g] -= self.LR* sum(self.layers[l].neurons[n].gradient[g])/ len(self.layers[l].neurons[n].gradient[g])
					self.layers[l].neurons[n].gradient[g] = []