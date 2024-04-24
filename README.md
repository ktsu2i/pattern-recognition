# Pattern recognition
This is a final project for CIS 3203: Introduction to Artificial Intelligence.

## Members
- Kaito Tsutsui
- Sahil Jartare

## Introduction
![Slide3](https://github.com/ktsu2i/pattern-recognition/assets/101069375/44f13eda-7f6a-4cdb-b317-1f17dce58149)
![Slide4](https://github.com/ktsu2i/pattern-recognition/assets/101069375/a47cf0d0-af1b-44d2-9df0-5b6f4c2eb6a4)

## Variables
### Input data
Each row of the input matrix below represents a single Hiragana character. Since we used 64 zeroes and ones to represent one Hiragana and gave 46 characters as input, the size of the matrix is 46x64.

$$
x =
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1\ 64}\\
x_{21} & x_{22} & \cdots & x_{2\ 64}\\
\vdots & \vdots & \ddots & \vdots\\
x_{46\ 1} & x_{46\ 2} & \cdots & x_{46\ 64}
\end{bmatrix}
$$

![input1](https://github.com/ktsu2i/pattern-recognition/assets/101069375/f46eaff5-5109-4b14-82d7-2f360da5dac4)![input2](https://github.com/ktsu2i/pattern-recognition/assets/101069375/527792b3-0ae7-435d-be86-ce7f898f850e)

### Expected output data
The expected output is a 46x46 identity matrix.

$$
y =
\begin{bmatrix}
1 & 0 & \cdots & 0\\
0 & 1 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & 1
\end{bmatrix}
$$

### Weights
To initialize all the weights, we used uniform distribution.

$w$ represents the weight matrix between input and hidden layer. The size is number of hidden nodes by number of each input data, which is 20x64 in this case.

$$
w =
\begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1~64}\\
w_{21} & w_{22} & \cdots & w_{2~64}\\
\vdots & \vdots & \ddots & \vdots\\
w_{20~1} & w_{20~2} & \cdots & w_{20~64}
\end{bmatrix}
$$

```
# w: NUM_OF_HIDDEN_NODES x Number of each input data
w = np.random.uniform(-0.5, 0.5, (NUM_OF_HIDDEN_NODES, x.shape[1]))
```

$b$ represents the bias matrix for the hidden nodes. The size is 1 by number of hidden nodes, which is 1x20 in this case.

$$
b =
\begin{bmatrix}
b_1 & b_2 & \cdots & b_{20}
\end{bmatrix}
$$

```
# b: 1 x NUM_OF_HIDDEN_NODES
b = np.random.uniform(-0.5, 0.5, NUM_OF_HIDDEN_NODES)
```

$u$ represents the weight matrix between hidden layer and output. The size is number of hidden nodes by number of inputs, which is 20x46 in this case.

$$
u =
\begin{bmatrix}
u_{11} & u_{12} & \cdots & u_{1~46}\\
u_{21} & u_{22} & \cdots & u_{2~46}\\
\vdots & \vdots & \ddots & \vdots\\
u_{20~1} & u_{20~2} & \cdots & u_{20~46}
\end{bmatrix}
$$

```
# u: NUM_OF_HIDDEN_NODES x NUM_OF_INPUTS
u = np.random.uniform(-0.5, 0.5, (NUM_OF_HIDDEN_NODES, NUM_OF_INPUTS))
```

$ub$ represents the bias matrix for the output. The size is 1 by number of inputs, which is 1x46 in  this case.

$$
ub =
\begin{bmatrix}
ub_1 & ub_2 & \cdots & ub_{46}
\end{bmatrix}
$$

```
# ub: 1 x NUM_OF_INPUTS
ub = np.random.uniform(-0.5, 0.5, NUM_OF_INPUTS)
```

## Learning process
We used 10000 iterations/epochs and 2 learning rate.

![Slide11](https://github.com/ktsu2i/pattern-recognition/assets/101069375/8d9b8b13-8c9e-4243-b055-0811192e842f)

### Hidden nodes for each row of input ($x_i$)

$$
\underset{1 \times 20}{hidden} =  sigmoid( \underset{1 \times 64}{x_i} \times \underset{64 \times 20}{w^T} + \underset{1 \times 20}{b} )
$$

### Output nodes for $x_i$

$$
\underset{1 \times 46}{output} =  sigmoid( \underset{1 \times 20}{hidden} \times \underset{20 \times 46}{u} + \underset{1 \times 46}{ub} )
$$

### Errors for each column of output ($y^T_j$)

$$
\underset{1 \times 46}{errors} = output - expected = \underset{1 \times 46}{output} - \underset{1 \times 46}{y^T_i}
$$

### Derivatives of errors
#### Sigmoid activation function

$$
F(x) = \sigma(x) = \frac{1}{1 + e^{-x}}
$$

#### Derivative of sigmoid activation function

$$
F'(x) = \sigma '(x) = \sigma (x) (1 - \sigma (x))
$$

Using the derivative of sigmoid function,

$$
\underset{1 \times 46}{dErrors} = \underset{1 \times 46}{errors} \odot sigmoidDerivative(\underset{1 \times 46}{output})
$$

```
def sigmoidDerivative(y):
    return y * (1 - y)
```

### Derivative of hidden nodes

$$
\underset{1 \times 46}{dHidden} = \underset{1 \times 46}{dErrors} \times \underset{46 \times 20}{u^T} \odot sigmoidDerivative(\underset{1 \times 20}{hidden})
$$

### Update weights

$$
\underset{20 \times 64}{w_{new}} = \underset{20 \times 64}{w_{old}} - \underset{1 \times 20}{dHidden} \otimes \underset{1 \times 64}{x_i} \odot \lambda
$$

$$
\underset{1 \times 20}{b_{new}} = \underset{1 \times 20}{b_{old}} - \underset{1 \times 20}{dHidden} \odot \lambda
$$

$$
\underset{20 \times 46}{u_{new}} = \underset{20 \times 46}{u_{old}} - \underset{1 \times 20}{hidden} \otimes \underset{1 \times 46}{dErrors} \odot \lambda
$$

$$
\underset{1 \times 46}{ub_{new}} = \underset{1 \times 46}{ub_{old}} - \underset{1 \times 46}{dErrors} \odot \lambda
$$

## Prediction for each row of input

$$
\underset{1 \times 20}{hidden} = sigmoid( \underset{1 \times 64}{x_i} \times \underset{64 \times 20}{w^T} + \underset{1 \times 20}{b})
$$

$$
\underset{1 \times 46}{output} = sigmoid( \underset{1 \times 20}{hidden} \times \underset{20 \times 46}{u} + \underset{1 \times 46}{ub})
$$

## Predicted output
All of the values on the diagonal line are close to 1, and others are close to 0. Hence, the neural network we built worked correctly.
![output](https://github.com/ktsu2i/pattern-recognition/assets/101069375/1a0fc744-29e1-4a98-941a-5a3a1c28c38c)

## Mean square errors
![plotting](https://github.com/ktsu2i/pattern-recognition/assets/101069375/1420119a-9148-45e3-a729-262c1243b56e)
