import math

# ===========CALL BY layer.Layer===========
# 
# 
class ActivationFunc:
	def __init__(self, use_func = None):
		self.activeFunc = use_func

	def feedForward(self, input):
		if self.activeFunc:
			result = {
				"sigmoid" : lambda input: 1/(1+ math.exp(-input))
			}[self.activeFunc](input)
		else:
			result = input
		return result

	def backForward(self, output):
		if self.activeFunc:
			result = {
				"sigmoid" : lambda output: output* (1- output)
			}[self.activeFunc](output)
		else:
			result = output
		return result

# ===========CALL BY Network===========
# 
# 
class Loss:
	def __init__(self, output, target, use_func = "SE"):
		self.output = output
		self.target = target
		self.loss = use_func
		self.defaultLoss = "self.squareError"
		self.defaultBP = "self.bp_squareError"
		index = {"SE": self.squareError}
		lossFunc = index.get(use_func, "Loss Function does not exist.")
		self.result = lossFunc(output, target)

	def __repr__(self):
		return str(sum(self.result))
	# feed forward
	def squareError(self, output, target):
		result = [(t-o)**2 / 2 for t,o in zip(target, output)]
		return result

	# back propagation
	def backForward(self):
		index = {"SE": self.bp_squareError}
		bpLossFunc = index.get(self.loss, "BP LOSS does not exist.")
		return bpLossFunc()

	def bp_squareError(self):
		result = [(o- t) for t,o in zip(self.target, self.output)]
		return result
