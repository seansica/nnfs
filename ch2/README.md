# Notes

**Fully connected**:
- every neuron in the current layer is connected to every neuron from the previous layer.
- very common type of neural network
- not required

Number of neurons determined by the number of sets of weights & biases. e.g., 3 sets of weights and 3 biases --> 3 neurons

TensorFlow --> comes from Tensors

Tensors:
- similar to arrays
- tensor/array/matrix - closely-related, diffs are subtle
- "A tensor object is an object that can be represented as an array"
    - What this means is, as programmers, we can (and will) treat tensors as arrays in the context of deep learning, and that’s really all the thought we have to put into it. Are all tensors just arrays? No, but they are represented as arrays in our code, so, to us, they’re only arrays, and this is why there’s so much argument and confusion.

We define an **array** as an ordered **homologous** container for numbers, and mostly use this term when working with the NumPy package 

Array == List (in Python)

Vector (in Math) == List (in Python) == 1D-array (in NumPy)

Matrix == rectangular or 2D array

e.g., a = [[4,2,3],[5,1]]
- Cannot be an array because it is not **homologous** (meaning each list along a given dimension is identically long)
    - first dimension's length is the parent array's length (2)
        - e.g., a.length
    - second dimension's length are the children's/element's array's length
        - e.g., a[0].length, a[1].length, ..., a[n].length

Dot Product = Used for vectors, i.e., vector multiplication

NN's tend to receive data in **batches**


To train, neural networks tend to receive data in **batches** So far, the example input data have been
only one sample (or **observation**) of various features called a **feature set**.

Sample == Observation == Feature Set

e.g., inputs = [1, 2, 3, 2.5]

Each of these values is a feature observation datum, and together they form a **feature set instance**, also called an **observation**, or most commonly, a **sample**.

We tend to train in batches because it is faster and improves generalization (or, in other words, reduces overfitting). If we feed samples to a neuron in serial, the algorithm will fit that that individual sample, rather than slowly producing generalized tweaks for the entire dataset.

Matrix product == 2D equivalent of vector product, e.g., high-school matrix multiplication!
- each elem of the output matrix is computed from the vector product of the row and the column (transposed). The intersection point is where the output elem is stored.
- np.dot(a, b.T)

**Transposition** simply modifies a matrix in a way that its rows become columns and columns become rows
- AKA rotate the matrix

