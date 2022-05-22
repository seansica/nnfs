# Notes

- Neural nets become "deep" when they have 2+ layers
- Data/samples flow -> input layer -> hidden layer -> output layer
- "dense" layers == "fully connected" == "fc"
- Weights are often initialized randomly for a model, but not always. Pre-trained models will load its weights to whatever that pre-trained model finished with.
- Forward pass: (not explicitely defined here but I think this is like backpropogation where we use the gradients to minimize loss)
- Why zero biases? In specific scenarios, like with many samples containing values of 0, a bias can ensure that a neuron fires initially. It sometimes may be appropriate to initialize the biases to some non-zero number, but the most common initialization for biases is 0.
- "Dead neurons" are mentioned
    - Dead neurons are specific to ReLU (I think...), specficically b/c of the fact that it produces a 1 when x > 0 and produces a 0 when x <= 0. So, what happens when we try to learn on examples where the activation IS zero? When we init the entire neural net to zero (and use ReLU), the zero'd neurons will always produce a 0 during forward propogatio, thus any gradient flowing through the neuron(s) thereafter will forever be 0 irrespective of the input.
    - Simply put, if all of our input samples flowing through our neurons happen to output a 0, we multiplicitively start producing 0 on other neurons b/c any weight multiplied by zero is zero (and our neurons calculate output by *weights x inputs + biases*).
- In general, neural nets work best with values between -1 and 1.
