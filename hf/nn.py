import math

# ===========CALL BY layer.Layer===========
# 
# 
class ActivationFunc:
	def __init__(self, use_func = None):
		self.activeFunc = use_func

	# def feedForward(self, input):
	# 	if self.activeFunc:
	# 		result = {
	# 			"sigmoid" : lambda input: 1/(1+ math.exp(-input))
	# 		}[self.activeFunc](input)
	# 	else:
	# 		result = input
	# 	return result

	# def backForward(self, output):
	# 	if self.activeFunc:
	# 		result = {
	# 			"sigmoid" : lambda output: output* (1- output)
	# 		}[self.activeFunc](output)
	# 	else:
	# 		result = output
	# 	return result

	def feedForward(self, neurons, input):
		index = {'sigmoid': self.sigmoid, 'softmax': self.softmax, 'relu': self.relu, }
		activeFunc = index.get(self.activeFunc, 'Active Function Not Exist.')
		return activeFunc(neurons, input)

	def sigmoid(self, neurons, input):
		result = 1/ (1+ math.exp(-neurons[input].input))
		return result

	def softmax(self, neurons, input):
		denominator = 0
		for neur in neurons:
			denominator += math.exp(neur.input)

		result = math.exp(neurons[input].input)/ denominator
		return result

	def relu(self, neurons, input):
		result = max(0, neurons[input].input)
		return result

	def backForward(self, neurons, output, targetMax):
		index = {'sigmoid': self.bp_sigmoid, 'softmax': self.bp_softmax, 'relu': self.bp_relu, }
		bpFunc = index.get(self.activeFunc, 'BP Function Not Exist.')
		if targetMax is None:
			return bpFunc(neurons, output)
		else:
			return bpFunc(neurons, output, targetMax)

	def bp_sigmoid(self, neurons, output):
		result = neurons[output].output* (1- neurons[output].output)
		return result

	def bp_softmax(self, neurons, output, targetMax):
		if output == targetMax:
			result = neurons[output].output* (1 - neurons[output].output)
		else:
			result = -neurons[output].output* neurons[targetMax].output
		return result

	def bp_relu(self, neurons, output):
		if neurons[output].output > 0:
			result = 1
		else:
			result = 0
		return result


# ===========CALL BY Network===========
# 
# 
class Loss:
	def __init__(self, output, target, use_func = "SE"):
		self.output = output
		self.target = target
		self.loss = use_func
		self.targetMax = target.index(max(target))

		index = {"SE": self.squareError, "CE": self.crossEntropy}
		lossFunc = index.get(use_func, "Loss Function does not exist.")
		self.result = lossFunc(output, target)
	
	def get(self):
		return sum(self.result)/ len(self.result)


	# feed forward
	def squareError(self, output, target):
		result = [(t-o)**2 / len(target) for t,o in zip(target, output)]
		return result

	def crossEntropy(self, output, target):
		result = [ -t* math.log(o) for t,o in zip(target, output)]
		return result

	# back propagation
	def backForward(self):
		index = {"SE": self.bp_squareError, "CE": self.bp_crossEntropy}
		bpLossFunc = index.get(self.loss, "BP LOSS does not exist.")
		return bpLossFunc()

	def bp_squareError(self):
		result = [(o- t) for t,o in zip(self.target, self.output)]
		return result

	def bp_crossEntropy(self):
		result = [ -1/ (self.output[self.targetMax]) for i in range(len(self.target))]
		return result
