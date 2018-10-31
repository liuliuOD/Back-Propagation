
class Optimizer:
	def __init__(self, neurons, use_func = "GD"):
		self.neurons = neurons
		
		index = {"GD": self.gradientDescent}
		optimizer = index.get(use_func, "Optimizer does not exist.")
		optimizer(neurons)

	def gradientDescent(self, neurons):
		for n in range(len(self.neurons)):
			for w in range(len(self.neurons[n].weight)):
				self.neurons[n].weight[w] -= self.neurons[n].gradient[w]
