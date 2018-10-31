import random
import hf.nn as nn
import hf.train as train

class Layer:
	def __init__(self, in_num, out_num, w = None, bias = 0, active = None):
		self.bias = bias
		self.neurons = []
		self.active = nn.ActivationFunc(active)
		for i in range(out_num):
			self.neurons.append(Neuron(bias))

		count = 0
		for o in range(out_num):
			for i in range(in_num):
				if w:
					self.neurons[o].weight.append(w[count])
					count += 1
				else:
					self.neurons[o].weight.append(random.random())

	def feedForward(self, input):
		self.input = input
		self.output = []
		for n in range(len(self.neurons)):
			self.neurons[n].input = 0
			for i in range(len(input)):
				self.neurons[n].input += self.neurons[n].weight[i]* input[i]
			
			self.neurons[n].input += self.bias
			self.neurons[n].output = self.active.feedForward(self.neurons[n].input)
			self.output.append(self.neurons[n].output)
		return self.output

	def backPropagation(self, derivation, opti_func = "GD"):
		gradient = []

		for n in range(len(self.neurons)):
			self.neurons[n].gradient = []
			self.neurons[n].derivation = self.active.backForward(self.neurons[n].output)* derivation[n]

		for i in range(len(self.input)):
			tmp = 0
			for neur in self.neurons:
				tmp += neur.derivation* neur.weight[i]
			gradient.append(tmp)


		for i in range(len(self.input)):
			for n in range(len(self.neurons)):
				self.neurons[n].gradient.append(self.neurons[n].derivation* self.input[i])
		
		train.Optimizer(self.neurons, opti_func)

		return gradient


# ===========CALL BY layer.Layer===========
# 
# 
class Neuron:
	def __init__(self, bias):
		self.bias = bias
		self.weight = []
		self.input = 0
		self.output = 0
		self.derivation = 0
		self.gradient = []