# Single Neuron

inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 1


# Our single neuron sums each input multiplied by that input’s weight, then adds the bias. 
# All the neuron does is take the fractions of inputs, where these fractions (weights) are the adjustable parameters, 
# and adds another adjustable parameter — the bias — then outputs the result. 



output = (inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias)
print(output)


#Output: 1.3


## Layer of Neurons
inputs = [1, 2, 4, 2.5]


weights1 = [0.1, 0.8, -0.5, 1.1]
weights2 = [0.5, -0.1, 0.6, -0.5]
weights3 = [-0.46, -0.27, 0.17, 0.8]
bias1 = 1
bias2 = 2
bias3 = 0.5
outputs = [
# Neuron 1:
inputs[0]*weights1[0] +
inputs[1]*weights1[1] +
inputs[2]*weights1[2] +
inputs[3]*weights1[3] + bias1,
# Neuron 2:
inputs[0]*weights2[0] +
inputs[1]*weights2[1] +
inputs[2]*weights2[2] +
inputs[3]*weights2[3] + bias2,
# Neuron 3:
inputs[0]*weights3[0] +
inputs[1]*weights3[1] +
inputs[2]*weights3[2] +
inputs[3]*weights3[3] + bias3
]


print(outputs)


# Output: [3.45, 3.4499999999999997, 2.18]


## Layer of Neuron's using a for loop
inputs = [1, 2, 4, 2.5]


weights = [[0.1, 0.8, -0.5, 1.1], [0.5, -0.1, 0.6, -0.5], [-0.46, -0.27, 0.17, 0.8]]
biases = [1, 2, 0.5]


# Output of current layer
layer_out = []
# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0
    # For each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
        # Multiply this input by the associated weight
        # and add to the neuron’s output variable
        neuron_output += n_input*weight
    # Adding bias term
    neuron_output += neuron_bias
    # Putting neuron’s result in the layer’s output list
    layer_out.append(neuron_output)
    print(layer_out)


print(layer_out[-1])


# Output: [3.45, 3.4499999999999997, 2.18]
