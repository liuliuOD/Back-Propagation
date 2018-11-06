import random
import hf.nn as nn

class Layer:
	def __init__(self, in_num, out_num, w = None, bias = None, active = None):
		self.neurons = []
		self.active = nn.ActivationFunc(active)
		for i in range(out_num):
			if bias != None:
				self.neurons.append(Neuron(bias[i]))
			else:
				self.neurons.append(Neuron(random.random()))

		count = 0
		for o in range(out_num):
			for i in range(in_num):
				self.neurons[o].gradient.append([])
				if w != None:
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
			self.neurons[n].input += self.neurons[n].bias
			

		for n in range(len(self.neurons)):
			self.neurons[n].output = self.active.feedForward(self.neurons, n)
			self.output.append(self.neurons[n].output)
		
		return self.output

	def backPropagation(self, derivation, targetMax = None):
		gradient = []
		# if targetMax:
		# 	for i in range(len(self.input)):
		# 		for n in range(len(self.neurons)):
		# 			if n == targetMax:
		# 				self.neurons[n].gradient[i].append((self.neurons[n].output - 1)* self.input[i])
		# 			else:
		# 				self.neurons[n].gradient[i].append((self.neurons[n].output)* self.input[i])
		# 	for i in range(len(self.input)):
		# 		tmp = 0
		# 		for n in range(len(self.neurons)):
		# 			if n == targetMax:
		# 				tmp += (self.neurons[n].output - 1)* self.neurons[n].weight[i]
		# 			else:
		# 				tmp += (self.neurons[n].output)* self.neurons[n].weight[i]
		# 		gradient.append(tmp)
		# else:
		for n in range(len(self.neurons)):
			
			# self.neurons[n].derivation = self.active.backForward(self.neurons, n, targetMax)* derivation[n]
			self.neurons[n].derivation.append(self.active.backForward(self.neurons, n, targetMax)* derivation[n])

		for i in range(len(self.input)):
			tmp = 0
			for neur in self.neurons:
				tmp += neur.derivation[-1]* neur.weight[i]
			gradient.append(tmp)


		for i in range(len(self.input)):
			for n in range(len(self.neurons)):
				self.neurons[n].gradient[i].append(self.neurons[n].derivation[-1]* self.input[i])
		
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
		self.derivation = []
		self.gradient = []