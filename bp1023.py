import math
import random

class NeuronNet:
	def __init__(self, num_each_layer = [2, 2, 2], hidden_weight = None, hidden_bias = 0,
			output_weight = None, output_bias = 0):
		self.hidden = Layer(num_each_layer[0], num_each_layer[1], hidden_weight, hidden_bias)
		self.output = Layer(num_each_layer[1], num_each_layer[2], output_weight, output_bias)
	
	def forWard(self, input):
		h1 = self.hidden.forWard(input)
		o1 = self.output.forWard(h1)
		# print("H1:{}, O1:{}".format(h1,o1))
		return o1

	# loss function = 0.5*sum((answer-input)**2)
	def loss(self, input, answer):
		# output = []
		# for i in range(len(input)):
		# 	output.append(0.5*(answer[i]- input[i])**2)			
		# return output
		output = 0
		for i in range(len(input)):
			output += 0.5*(answer[i]- input[i])**2
		return output

	def backWard(self, input, answer):
		# ∂L/∂O
		dL = self.bpLoss(input, answer)
		# ∂O/∂netO
		dO = self.output.bpActiv()
		# ∂L/∂netO
		dLnetO = [a*b for a,b in zip(dL, dO)]
		# ∂H/∂netH
		dH = self.hidden.bpActiv()
		# ∂L/∂H
		dLH = self.hidden.bpHidden(self.output.neurons, dLnetO)
		# ∂L/∂netH
		dLnetH = [a*b for a,b in zip(dH, dLH)]
		# ∂netO/∂Wo
		dnetO = self.output.weightGD(dLnetO)
		# ∂netH/∂Wh
		dnetH = self.hidden.weightGD(dLnetH)

		# weight update
		self.output.optimizer(dnetO)
		self.hidden.optimizer(dnetH)

	def bpLoss(self, input, answer):
		output = []
		for i in range(len(input)):
			output.append(input[i]- answer[i])
		return output

	def train(self, input, answer):
		predict = self.forWard(input)
		loss = self.loss(predict, answer)
		self.backWard(predict, answer)
		print("Loss :{}\nPredict :{}".format(loss, predict))




class Layer:
	def __init__(self, input_num, neuron_num, weight = None, bias = 0):
		# self.neurons = [Neuron(bias)] * neuron_num 錯誤寫法，會造成同時對多個相同位址的物件輸入
		self.bias = bias
		self.neurons = []
		for i in range(neuron_num):
			self.neurons.append(Neuron(bias))

		count_weight = 0
		for i in range(len(self.neurons)):
			for j in range(input_num):
				if weight:
					self.neurons[i].weight.append(weight[count_weight])
					count_weight += 1
				else:
					self.neurons[i].weight.append(random.random())

	def forWard(self, input):
		# save for back propagation
		self.input = input
		self.output = []
		for i in range(len(self.neurons)):
			result = 0
			for j in range(len(input)):
				result += input[j]* self.neurons[i].weight[j]
			self.output.append(self.activeFunc(result+ self.bias))
		return self.output

	# sigmoid
	def activeFunc(self, input):
		return 1/ (1+ math.exp(-input))

	# sigmoid* (1- sigmoid)
	def bpActiv(self):
		output = []
		for o in self.output:
			output.append(o* (1-o))
		return output

	def bpHidden(self, neurons, dX):
		output = []
		for i in range(len(self.neurons)):
			result = 0
			for neuron in neurons:
				result += neuron.weight[i]* dX[i]
			output.append(result)
		return output

	def weightGD(self, dX):
		output = []
		for diff in dX:
			for input in self.input:
				output.append(input* diff)
		return output
				

	def optimizer(self, gradient, LR = 0.5):
		for i in range(len(self.neurons)):
			for j in range(len(self.neurons[i].weight)):
				self.neurons[i].weight[j] -= LR* gradient[i* len(self.neurons[i].weight)+ j]


class Neuron:
	def __init__(self, bias):
		self.bias = bias
		self.weight = []



if __name__ == '__main__':
	nn = NeuronNet(hidden_weight = [0.15, 0.2, 0.25, 0.3],
	 hidden_bias = 0.35, output_weight = [0.4, 0.45, 0.5, 0.55], output_bias = 0.6)
	for i in range(2):
		nn.train([0.05, 0.1], [0.01, 0.09])
