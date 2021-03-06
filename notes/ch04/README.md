# Notes

- Activation fn's are applied to the output of a neuron (or the output layer)
- They allow hidden layers to map nonlinear fn's
- Usually, neural networks use two types of activation fn's:
    1. those used in hidden layers
    2. those used in the output layer
- **Step Activation Function**
    - simple binary function; "fires" a 1 if output > 0, otherwise produces a 0
    - rarely used anymore
- **Linear Activation Function**
    - usually applied to the output layer in regression models
    - regression models output scalar values rather than classifications
- **Sigmoid Activation Function**
    - more granular approach to binary approach, i.e. Step Function
    - allows us to output values between 0 and 1, thereby allowing our model to more accurately correct for loss
    - returns 0-0.5 for values -math.inf to 0
    - returns 0.5-1 for values 1 to math.inf
- Usually it is better to use a more granular activation fn in your hidden layer(s)
- **Rectified Linear Activation Function (ReLU)**
    - effectively replaced Sigmoid
    - y=x, clipped at 0
    - y = 0 if x <=0 else x
    - "If x is less than or equal to 0, then y is 0 — otherwise, y is equal to x."
    - currently the most popular activation fn
    - Bias offsets the activation point *horizontally*, i.e. by increasing bias, we make the neuron activate earlier
    - Weight influences the slope of the activation
    - Weight polarity (negative vs positive weight) determines the direction of the graphed fn
        - Negative weight will yield the neuron *deactivation* point
        - The point at which y=x (i.e. we cross the x-axis) is the point at which the fn deactivates when weight is negative
    - Is unbounded, not normalized, and exclusive
        - "not normalized" means the values can be anything; they are w/o context
        - exclusive" means each output is independent of the others
- Softmax Activation Function
    - Used for classification
    - Can take in non-normalized, or uncalibrated inputs and produce a normalized distribution of probabilities for our classes
    - In other words, it produce **confidence scores** for each class. All scores in the scalar output will sum to 1. The predicted class is associated with whichever neuron returned the largest score.
    - To calculate probabilities, we need non-negative numbers. That is why we exponentiate them in Softmax fn. An exponentiated value is always positive.
    - There are two main pervasive challenges with neural networks: “dead neurons” and very large numbers (referred to as “exploding” values).
    - The exponential function used in softmax activation is one of the sources of exploding values. Consider, for example, an input value of 1000. Exponentiating 1000 (e ** 1000) would cause an overflow error.
