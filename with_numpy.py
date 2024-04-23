import random
import numpy as np
import matplotlib.pyplot as plt


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDerivative(y):
    return y * (1 - y)


def backpropagationWithNumpy(x, y, iterations, learningRate):
    NUM_OF_HIDDEN_NODES = 20
    
    NUM_OF_INPUTS = len(x)
    
    # w: NUM_OF_HIDDEN_NODES x Number of each input data
    w = np.random.uniform(-0.5, 0.5, (NUM_OF_HIDDEN_NODES, x.shape[1]))
    
    # b: 1 x NUM_OF_HIDDEN_NODES
    b = np.random.uniform(-0.5, 0.5, NUM_OF_HIDDEN_NODES)
    
    # u: NUM_OF_HIDDEN_NODES x NUM_OF_INPUTS
    u = np.random.uniform(-0.5, 0.5, (NUM_OF_HIDDEN_NODES, NUM_OF_INPUTS))
    
    # ub: 1 x NUM_OF_INPUTS
    ub = np.random.uniform(-0.5, 0.5, NUM_OF_INPUTS)
    
    # Mean square errors
    meanSquareErrors = []

    # Epoch (iterations)
    for i in range(0, iterations):
        total_error = 0
        
        # Each row of x
        for j in range(0, NUM_OF_INPUTS):
            # hidden: 1 x NUM_OF_HIDDEN_NODES
            hidden = sigmoid_np(x[j] @ w.T + b)
            
            # output: 1 x NUM_OF_INPUTS
            output = sigmoid_np(hidden @ u + ub)
            
            # errors: 1 x NUM_OF_INPUTS
            errors = output - y.T[j]
            
            # dErrors: 1 x NUM_OF_INPUTS
            dErrors = errors * sigmoidDerivative(output)
            
            # Add average mean square error to total
            total_error += np.square(errors).sum() / NUM_OF_INPUTS
            
            
            # dHidden: 1 x NUM_OF_HIDDEN_NODES
            dHidden = dErrors @ u.T * sigmoidDerivative(hidden)
            
            # Update weights
            w -= np.outer(dHidden, x[j]) * learningRate
            b -= dHidden * learningRate
            u -= np.outer(hidden, dErrors) * learningRate
            ub -= dErrors * learningRate
            
        meanSquareErrors.append(total_error / NUM_OF_INPUTS)
        
    # Plot       
    plt.plot(meanSquareErrors)
    plt.title("Final Project")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Square Error")
    plt.show()
            
    # Predict
    predicted = np.zeros((NUM_OF_INPUTS, NUM_OF_INPUTS))
        
    for j in range(0, NUM_OF_INPUTS):
        hidden_layer = sigmoid_np(x[j] @ w.T + b)
        predicted_output = sigmoid_np(hidden_layer @ u + ub)
        predicted[j] = predicted_output
            
    return predicted.T


# main
              
'''
8x8 canvas
[0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0],
'''

x = np.array([[0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 1, 0,
               0, 0, 1, 1, 1, 1, 0, 0,
               0, 1, 0, 1, 0, 1, 1, 0,
               0, 1, 0, 1, 1, 0, 1, 0,
               0, 0, 1, 0, 0, 0, 1, 0],
              
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 1, 0, 0,
               0, 1, 0, 0, 0, 0, 1, 0,
               0, 1, 0, 1, 0, 0, 1, 0,
               0, 1, 0, 1, 0, 0, 1, 0,
               0, 0, 1, 0, 0, 0, 0, 0],
              
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 1, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 1, 1, 0, 0, 0,
               0, 1, 0, 0, 0, 1, 0, 0,
               0, 0, 0, 0, 0, 1, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 0, 1, 1, 0, 0, 0, 0],
              
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 1, 1, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 1, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 1, 0, 1, 0, 0, 0,
               0, 1, 0, 0, 0, 1, 1, 0],
              
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 0, 1, 0, 0,
               0, 0, 1, 0, 0, 0, 1, 0,
               0, 0, 1, 1, 1, 1, 0, 0,
               0, 1, 1, 0, 0, 0, 1, 0,
               1, 0, 1, 0, 0, 0, 1, 0,
               0, 1, 1, 0, 1, 1, 0, 0],
              
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 1, 0,
               0, 1, 1, 1, 1, 0, 0, 1,
               0, 0, 1, 0, 0, 1, 0, 1,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 1, 0, 0, 0, 1, 0, 0,
               0, 0, 0, 1, 1, 0, 0, 0],
              
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 1, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 1, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 1, 1, 0, 0, 0],
              
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0],
              
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 1, 0, 0,
               0, 1, 0, 0, 0, 1, 0, 0,
               0, 1, 0, 1, 1, 1, 1, 0,
               0, 1, 0, 0, 0, 1, 0, 0,
               0, 1, 0, 0, 0, 1, 0, 0,
               0, 1, 1, 0, 0, 1, 0, 0,
               0, 1, 0, 0, 1, 0, 0, 0],
              
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 1, 0, 0,
               0, 0, 0, 0, 0, 1, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 1, 1, 1, 0, 0],
              
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 1, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 0, 1, 1, 1, 0, 0, 0,
               0, 1, 0, 0, 0, 1, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 1, 1, 0, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 0, 0, 1, 1, 0, 0, 0],
              
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 1, 1, 1, 1, 1, 0, 0,
               0, 0, 0, 1, 1, 0, 0, 0,
               0, 0, 1, 0, 1, 0, 0, 0,
               0, 0, 0, 1, 1, 0, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 0, 1, 1, 0, 0, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 1, 1, 1, 1, 1, 1, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 0, 1, 0, 1, 1, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 1, 1, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 1, 1, 1, 0, 0,
               0, 0, 0, 0, 0, 1, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 0, 1, 1, 1, 1, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 0, 1, 1, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 1, 1, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 1, 0, 0, 0, 0,
               0, 1, 0, 0, 1, 1, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 1, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 0, 1, 1, 1, 1, 0, 0,
               0, 0, 1, 0, 0, 0, 1, 0,
               0, 0, 0, 0, 0, 0, 1, 0,
               0, 0, 0, 1, 1, 1, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 1, 1, 0, 0,
               0, 1, 1, 0, 0, 0, 1, 0,
               0, 0, 0, 0, 0, 0, 1, 0,
               0, 0, 0, 0, 0, 0, 1, 0,
               0, 0, 0, 0, 0, 1, 0, 0,
               0, 0, 0, 1, 1, 0, 0, 0],
                          
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 1, 1, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 0, 1, 1, 1, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 1, 1, 0, 0,
               0, 0, 1, 1, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 1, 1, 1, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 0, 1, 0, 0,
               0, 0, 1, 0, 0, 0, 1, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 1, 0, 0, 1, 1, 0, 0,
               0, 1, 0, 1, 0, 1, 0, 0,
               0, 0, 0, 1, 1, 0, 1, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 1, 1, 1, 1, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 1, 0, 0, 0, 0,
               0, 1, 0, 0, 1, 1, 1, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               1, 0, 1, 1, 1, 0, 0, 0,
               1, 1, 0, 1, 0, 1, 0, 0,
               0, 1, 0, 1, 0, 1, 0, 0,
               1, 0, 1, 0, 0, 1, 0, 0,
               1, 0, 1, 0, 1, 0, 1, 0,
               0, 1, 0, 0, 1, 1, 0, 1],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 1, 1, 0, 1, 1, 0, 0,
               0, 0, 1, 1, 0, 0, 1, 0,
               0, 0, 1, 0, 0, 0, 1, 0,
               0, 1, 1, 0, 0, 1, 1, 0,
               0, 1, 1, 0, 1, 0, 1, 0,
               0, 0, 1, 0, 1, 1, 0, 1],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 1, 1, 0, 0, 0,
               0, 1, 0, 1, 0, 1, 0, 0,
               1, 0, 0, 1, 0, 0, 1, 0,
               1, 0, 1, 0, 0, 0, 1, 0,
               1, 0, 1, 0, 0, 0, 1, 0,
               0, 1,  0, 0, 1, 1, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 1, 0, 0,
               0, 1, 0, 1, 1, 1, 1, 0,
               0, 1, 0, 0, 0, 1, 0, 0,
               0, 1, 0, 0, 0, 1, 0, 0,
               0, 1, 0, 0, 1, 1, 0, 0,
               0, 1, 0, 1, 0, 1, 1, 0,
               0, 1, 0, 0, 1, 0, 0, 1],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 1, 1, 0, 0, 1, 0, 0,
               0, 0, 1, 0, 0, 1, 1, 0,
               0, 1, 0, 0, 0, 1, 0, 1,
               0, 1, 0, 0, 0, 1, 0, 0,
               0, 1, 0, 0, 0, 1, 0, 0,
               0, 0, 1, 1, 1, 0, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 1, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 1, 0, 1, 0, 1, 0, 0,
               0, 1, 0, 0, 1, 0, 1, 0,
               1, 0, 0, 0, 1, 0, 1, 0,
               0, 0, 1, 1, 0, 0, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 1, 0, 0, 0, 0,
               0, 1, 0, 0, 1, 0, 0, 0,
               1, 0, 0, 0, 0, 1, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 0,
               0, 0, 0, 0, 0, 0, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               1, 0, 1, 1, 1, 1, 0, 0,
               1, 0, 0, 0, 1, 0, 0, 0,
               1, 0, 1, 1, 1, 1, 1, 0,
               1, 0, 0, 0, 1, 0, 0, 0,
               1, 0, 0, 1, 1, 0, 0, 0,
               1, 0, 1, 0, 1, 1, 0, 0,
               1, 0, 0, 1, 1, 0, 1, 0],
                            
              [0, 0, 0, 1, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 1, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 1, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 1, 1, 1, 0, 0, 0,
               0, 1, 0, 1, 0, 1, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 1, 0, 0,
               0, 1, 1, 1, 1, 1, 0, 0,
               1, 0, 1, 0, 0, 1, 1, 0,
               1, 0, 1, 0, 0, 1, 0, 0,
               0, 1, 0, 0, 1, 0, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 0, 1, 0,
               0, 0, 1, 0, 0, 0, 0, 1,
               0, 1, 1, 0, 0, 0, 0, 0,
               1, 0, 1, 0, 0, 0, 1, 0,
               0, 1, 1, 0, 0, 0, 1, 0,
               0, 0, 1, 1, 1, 1, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               1, 0, 1, 1, 1, 0, 0, 0,
               1, 1, 0, 1, 0, 1, 0, 0,
               0, 1, 0, 1, 0, 0, 1, 0,
               1, 0, 1, 0, 0, 0, 1, 0,
               1, 0, 1, 1, 0, 0, 1, 0,
               0, 1, 0, 0, 0, 1, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 0, 0, 1, 0,
               0, 0, 1, 0, 0, 0, 1, 0,
               0, 0, 0, 1, 1, 1, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 1, 0, 0, 0,
               0, 1, 0, 1, 1, 1, 0, 0,
               0, 1, 1, 0, 1, 0, 1, 0,
               1, 0, 1, 0, 0, 0, 1, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 1, 0, 1, 1, 1, 0, 0,
               0, 1, 1, 0, 1, 0, 1, 0,
               0, 1, 0, 0, 1, 0, 1, 0,
               0, 1, 0, 1, 1, 0, 1, 0,
               0, 1, 0, 0, 1, 1, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 0, 0, 0, 1, 1, 1, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 0, 0, 1, 1, 0, 0, 0,
               0, 0, 1, 0, 1, 1, 0, 0,
               0, 0, 0, 1, 0, 0, 1, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 1, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 1, 1, 0, 0, 0,
               0, 1, 1, 0, 0, 1, 0, 0,
               0, 0, 0, 0, 0, 1, 0, 0,
               0, 0, 0, 1, 1, 0, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 0, 0, 1, 0, 1, 0, 0,
               0, 0, 0, 0, 0, 1, 0, 0,
               0, 0, 0, 1, 1, 0, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 1, 1, 1, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 0, 0, 1, 1, 1, 0, 0,
               0, 0, 1, 0, 0, 0, 1, 0,
               0, 0, 0, 0, 1, 0, 1, 0,
               0, 0, 0, 1, 0, 1, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 1, 1, 0, 0,
               0, 1, 1, 1, 0, 1, 0, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 0, 1, 0, 0, 1, 0, 0,
               0, 1, 1, 0, 0, 1, 0, 0,
               0, 0, 1, 0, 0, 1, 0, 1,
               0, 0, 1, 0, 0, 0, 1, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 1, 1, 1, 0, 0,
               0, 0, 0, 0, 1, 0, 0, 0,
               0, 0, 0, 1, 1, 1, 0, 0,
               0, 0, 1, 0, 0, 0, 1, 0,
               0, 0, 0, 0, 0, 0, 1, 0,
               0, 0, 0, 0, 0, 0, 1, 0,
               0, 0, 0, 1, 1, 1, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 1, 1, 0, 0,
               0, 1, 1, 1, 0, 0, 1, 0,
               0, 0, 1, 0, 0, 0, 1, 0,
               0, 1, 1, 0, 0, 0, 1, 0,
               0, 0, 1, 0, 0, 0, 1, 0,
               0, 0, 1, 0, 1, 1, 0, 0],
                            
              [0, 0, 0, 1, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 0, 0, 0,
               0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 1, 1, 0,
               0, 1, 0, 1, 1, 0, 0, 0,
               0, 0, 1, 0, 1, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 1, 1, 0, 0],
                            
              [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0,
               0, 1, 1, 0, 0, 0, 0, 0,
               0, 1, 0, 1, 0, 0, 1, 0,
               1, 0, 0, 1, 0, 0, 1, 0,
               1, 0, 0, 0, 1, 1, 0, 0],])

y = np.identity(46)

predicted = backpropagationWithNumpy(x, y, 10000, 2)

print("Predicted Output:")
np.set_printoptions(threshold = np.inf, linewidth = np.inf)
print(np.around(predicted, 3))