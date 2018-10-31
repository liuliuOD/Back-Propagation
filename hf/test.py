import nn
import layer
# test = nn.ActivationFunc(None)
# print(test.forWard(0.8))

# test = layer.Layer(2, 4, [1.0, 1.0, 2.0, 2.0, 3., 3.0, 4.0, 4.0]);
# for n in test.neurons:
# 	print(n.weight)
test1 = layer.Layer(2, 2, [0.15, 0.2, 0.25, 0.3], 0.35, active = "sigmoid")
test2 = layer.Layer(2, 2, [0.4, 0.45, 0.5, 0.55], 0.6, active = "sigmoid")
for i in range(1000):
	h1 = test1.feedForward([0.05, 0.1])
	o1 = test2.feedForward(h1)
	loss = nn.Loss(o1, [0.01, 0.09])
	gradient = loss.backForward()
	gradient = test2.backPropagation(gradient, isHidden = False)
	gradient = test1.backPropagation(gradient)
	print("LOSS :{}".format(loss))
	print(o1)