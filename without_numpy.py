import random
import math
import matplotlib.pyplot as plt
import numpy as np


def printMatrix(a_matrix):
    nRow = len(a_matrix)
    nCol = len(a_matrix[0])
    
    for i in range(0, nRow):
        for j in range(0, nCol):
            print("{:.5f}".format(a_matrix[i][j]), end = " ")
        print()


def sigmoid(x):
    return 1 / (1 + math.e ** (-x))


def sigmoidDerivative(y):
    return y * (1 - y)


# Without numpy
def backpropagation(x, y, iterations, learningRate):
    NUM_OF_HIDDEN_NODES = 5
    NUM_OF_INPUTS = len(x)
    
    # initialize weights w: 5 x 25
    w = []
    for i in range(0, NUM_OF_HIDDEN_NODES):
        row = []
        for j in range(0, len(x[0])):
            value = random.uniform(-0.5, 0.5)
            row.append(value)
        w.append(row)
        
    # b values [b1, b2, b3, b4, b5]
    b = []
    for i in range(0, NUM_OF_HIDDEN_NODES):
        b.append(random.uniform(-0.5, 0.5))
       
    # weights from hidden layer: 5 x 3
    u = []
    for i in range(0, NUM_OF_HIDDEN_NODES):
        row = []
        for i in range(0, NUM_OF_INPUTS):
            value = random.uniform(-0.5, 0.5)
            row.append(value)
        u.append(row)
        
    # ub: [ub1, ub2, ub3]
    ub = []
    for i in range(0, NUM_OF_INPUTS):
        ub.append(random.uniform(-0.5, 0.5))
        
    #meanSquareErrors = []
    #xCoordinates = []
    
    for i in range(0, iterations):
        #meanSquareError = 0
        
        # Each row of x
        for j in range(0, NUM_OF_INPUTS):
            hidden = []
        
            # Calculate hidden layer
            for k in range(0, NUM_OF_HIDDEN_NODES):
                h = b[k]
            
                for l in range(0, len(x[0])):
                    h += w[k][l] * x[j][l]
                
                hidden.append(sigmoid(h))
                
            # Calculate output = sigmoid(hidden * u)
            output = []
            for k in range(0, NUM_OF_INPUTS):
                output.append(ub[k])
                
            for k in range(0, NUM_OF_INPUTS):
                for l in range(0, NUM_OF_HIDDEN_NODES):
                    output[k] += hidden[l] * u[l][k]
                
            for k in range(0, NUM_OF_INPUTS):
                output[k] = sigmoid(output[k])

            # Calculate errors
            errors = []
            #meanSquareError = 0
            for k in range(0, NUM_OF_INPUTS):
                errors.append(output[k] - y[j][k])
                #meanSquareError += (output[k] - y[j][k]) ** 2
                
            #meanSquareError /= NUM_OF_INPUTS            
            
            # Derivatives of error
            dErrors = []
            for k in range(0, NUM_OF_INPUTS):
                dErrors.append(errors[k] * sigmoidDerivative(output[k]))
            
            # Derivatives of hidden nodes
            dHidden = []
            for k in range(0, NUM_OF_HIDDEN_NODES):
                value = 0
                for l in range(0, NUM_OF_INPUTS):
                    value += dErrors[l] * u[k][l]
                dHidden.append(value * sigmoidDerivative(hidden[k]))
            
            # Update weights
            for k in range(0, NUM_OF_INPUTS):
                ub[k] -= learningRate * dErrors[k]
            
            for k in range(0, NUM_OF_HIDDEN_NODES):
                # Update u
                for l in range(0, NUM_OF_INPUTS):
                    u[k][l] -= learningRate * hidden[k] * dErrors[l]
                
                # Update b
                b[k] -= learningRate * dHidden[k]
                
                # Update w
                for l in range(0, len(w[0])):
                    w[k][l] -= learningRate * x[j][l] * dHidden[k]
                    
        #meanSquareErrors.append(meanSquareError / len(x))
    '''
    for i in range(1, len(meanSquareErrors) + 1):
        xCoordinates.append(i)
    
    
    plt.plot(xCoordinates, meanSquareErrors)
    plt.title("XOR Gate")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Square Errors")
    plt.show()
    '''
    
    # Calculate predicted y
    predicted = []

    # Each row of x
    for i in range(0, NUM_OF_INPUTS):
        hiddenLayer = []
        
        # Each hidden node
        for j in range(0, NUM_OF_HIDDEN_NODES):
            h = b[j]
            
            for k in range(0, len(x[0])):
                h += w[j][k] * x[i][k]
                
            hiddenLayer.append(sigmoid(h))
        
        # Output: o1, o2, o3
        output = []
        for j in range(0, NUM_OF_INPUTS):
            output.append(ub[j])
            
        for j in range(0, NUM_OF_INPUTS):
            for k in range(0, NUM_OF_HIDDEN_NODES):
                output[j] += hiddenLayer[k] * u[k][j]
            output[j] = sigmoid(output[j])
            
        predicted.append(output)

    return predicted


# main
x = [[0, 0, 1, 0, 0,
      0, 1, 1, 0, 0,
      0, 0, 1, 0, 0,
      0, 0, 1, 0, 0,
      0, 1, 1, 1, 0],
     
     [0, 1, 1, 1, 0,
      0, 0, 0, 1, 0,
      0, 1, 1, 1, 0,
      0, 1, 0, 0, 0,
      0, 1, 1, 1, 0],
     
     [0, 1, 1, 1, 0,
      0, 0, 0, 1, 0,
      0, 1, 1, 1, 0,
      0, 0, 0, 1, 0,
      0, 1, 1, 1, 0],
     
     [0, 1, 0, 1, 0,
      0, 1, 0, 1, 0,
      0, 1, 1, 1, 0,
      0, 0, 0, 1, 0,
      0, 0, 0, 1, 0],
     
     [0, 1, 1, 1, 0,
      0, 1, 0, 0, 0,
      0, 1, 1, 1, 0,
      0, 0, 0, 1, 0,
      0, 1, 1, 1, 0],]

y = [[1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1]]

predicted = backpropagation(x, y, 10000, 2)

print("Predicted Output:")
printMatrix(predicted)