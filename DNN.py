import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer=0, bias_regularizer=0):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons), dtype='float32')
        self.weight_regularizer = weight_regularizer
        self.bias_regularizer = bias_regularizer

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer > 0:
            self.dweights += 2 * self.weight_regularizer * self.weights

        if self.bias_regularizer > 0:
            self.dbiases += 2 * self.bias_regularizer * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)
    
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        # The trick of subtracting the maximum value before exponentiation helps maintain numerical stability.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
        # Actual softmax formula after subtracting (substract won't affect final output since both dinominator and numerato are scaled simily.
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        self.sample_losses = self.forward(output, y)
        data_loss = np.mean(self.sample_losses) # mean loss of all the samples in the batch
        return data_loss
    
    def regularization_loss(self, layer: Layer_Dense):
        regularization_loss = 0

        if layer.weight_regularizer > 0:
            regularization_loss += layer.weight_regularizer * np.sum(layer.weights * layer.weights)
        if layer.bias_regularizer > 0:
            regularization_loss += layer.bias_regularizer * np.sum(layer.biases * layer.biases)

        return regularization_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # If all neurons output after the softmax is zero then log(0) will undifined, therefore we add a small value to it

        # Formula is -sum(y(t)*log(y(p)))
        # Since y_true is 1 or 0, we can reduce the formula as below and then get the log
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # If y_true given as one hot encoded(2D) then it converted into a list
        # Ex: [[1, 0, 0], [0, 1, 0]] = [0, 1]
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples # Not cleared

class Optimzer_SGD:
    def __init__(self, learning_rate=1, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learining_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learining_rate = self.learning_rate / (1. + self.decay * self.iterations)
    
    def update_params(self, layer: Layer_Dense):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learining_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learining_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.learning_rate * layer.dweights
            bias_updates = -self.learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

class Optimzer_ADAGRAD:
    def __init__(self, learning_rate=1, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learining_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon # Use to prevent division by zero error

    def pre_update_params(self):
        if self.decay:
            self.current_learining_rate = self.learning_rate / (1. + self.decay * self.iterations)
    
    def update_params(self, layer: Layer_Dense):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learining_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learining_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimzer_RMSProp:
    def __init__(self, learning_rate=1, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learining_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon # Use to prevent division by zero error
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learining_rate = self.learning_rate / (1. + self.decay * self.iterations)
    
    def update_params(self, layer: Layer_Dense):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases**2

        layer.weights += -self.current_learining_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learining_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimzer_Adam:
    def __init__(self, learning_rate=1, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learining_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon # Use to prevent division by zero error
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learining_rate = self.learning_rate / (1. + self.decay * self.iterations)
    
    def update_params(self, layer: Layer_Dense):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1-self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1-self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1-self.beta_1 ** (self.iterations + 1))
        bias_moemntums_corrected = layer.bias_momentums / (1-self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1-self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1-self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learining_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learining_rate * bias_moemntums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


X, y = spiral_data(samples = 100, classes=3)

dense1 = Layer_Dense(2, 64, weight_regularizer=5e-4, bias_regularizer=5e-4)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 3, weight_regularizer=5e-4, bias_regularizer=5e-4)
activation2 = Activation_Softmax()

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimzer = Optimzer_Adam(learning_rate=0.02, decay=1e-7)

for epoch in range(10000):
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    loss += regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' + 
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' + 
              f'lr: {optimzer.current_learining_rate:.3f}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimzer.pre_update_params()
    optimzer.update_params(dense1)
    optimzer.update_params(dense2)
    optimzer.post_update_params()

# Validation
X_test, y_test = spiral_data(samples=100, classes=3)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')