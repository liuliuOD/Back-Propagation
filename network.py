import hf.layer
import hf.nn

class Network:
	def __init__(self, optimizer = "GD"):
		self.h1 = hf.layer.Layer(4, 2, bias = 0.35, active = "sigmoid")
		self.h2 = hf.layer.Layer(2, 4, active = "sigmoid")
		self.h3 = hf.layer.Layer(4, 4, active = "sigmoid")
		self.o1 = hf.layer.Layer(4, 4, bias = 0.6, active = "sigmoid")
		self.optimizer = optimizer

	def train(self, input, target, epoch = 10):
		for i in range(epoch):
			h1 = self.h1.feedForward(input)
			h2 = self.h2.feedForward(h1)
			h3 = self.h3.feedForward(h2)
			o1 = self.o1.feedForward(h3)
			loss = hf.nn.Loss(o1, target)

			gradient = loss.backForward()
			gradient = self.o1.backPropagation(gradient, self.optimizer)
			gradient = self.h3.backPropagation(gradient, self.optimizer)
			gradient = self.h2.backPropagation(gradient, self.optimizer)
			gradient = self.h1.backPropagation(gradient, self.optimizer)
			print("LOSS [{}] :{}".format(i, loss))
			print("Predict [{}]:{}".format(i, o1))

if __name__ == '__main__':
	train = Network()
	train.train([0.1, 0.2, 0.3], [1, 0, 0], epoch = 10000)