# Notes

Using Python 3.7, NumPy, Matplotlib

Dense layers == series of fully interconnected neurons where each all neurons of a given layer connect to all neurons in the subsequent layer

### Neurons:
- *Output = weight · input + bias*
- Weight:
    - Each connection between neurons has a *weight* associated with it.
    - The weight is trainable. It specifies how how of this input to use. 
    - Weight is multiplied by the *input*.
- Bias:
    - Once all of the *inputs x weights* flow into a neuron, they are summed, and a *bias*, another trainable parameter, is added.
    - The bias lets us offset the final output + or -
- Weights and biases are like "knobs" that we can tune to fit our model


### Activation Functions:

Simple example of activation fn = **step funtion**:
- Mimics a neuron in the brain, binary on or off
- If output > 0 --> "fire" / output or return 1
- If output < 0 --> output "no fire" / output or return 0
- Named "step function" because the visual graphed function looks like a step

Neural networks of today tend to use more informative activation functions (rather than a step function), such as the Rectified Linear (ReLU) activation function.


### Layers:
- Input layer
    - represents actual input data like data from a temp. sensor
    - Usually we **preprocess** input data through functions like **normalization** and **scaling**.
    - Input data always *numerical*.
- Output layer
    - whatever the NN returns, e.g., classification output might be a binary output "dog":"cat" or "dog":"not dog"
- Hidden layer(s)


**Parameters** = weights + biases (a general way of describing the "things" that we modify to influence the output of a NN)

The end goal for neural networks is to adjust their weights and biases (the parameters), so when applied to a yet-unseen example in the input, they produce the desired output.

Overfitting occurs when we expose too much training data (labeled data?), thus resulting in the function "memorizing" the training data. i.e. The algorithm only learns to fit the training data but fails to adapt to other sample sets

To avoid overfitting we use “in-sample” data to train a model and then use “out-of-sample” data to validate an algorithm. This is called **generalization**, which means learning to fit the data instead of memorizing it.

*Loss* or error = a measurement of how "wrong" the output of a model is. Loss is factored into subsequent inputs to iteratively decrease loss.