import sys
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

#########################################
NUM_ITERS = 200 ####Do not edit this!####
#########################################

if __name__ == "__main__":
    #Read in inputs
    filename = sys.argv[1]
    df = pd.read_csv(filename)

    #Question 1 - Visualize Data
    plt.figure()
    plt.plot(df['year'],df['days'])
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.savefig("data_plot.jpg")


    #Question 2 - Data Normalization
    print("Q2:")
    array = np.array(df)

    min = array.min(axis = 0)
    max = array.max(axis = 0)

    X_set = (array[:, 0] - min[0])/ (max[0] - min[0])
    ones_vector = np.ones((array.shape[0], 1))
    X_normalized = np.column_stack([X_set, ones_vector])

    print(X_normalized)

    
    #Question 3 - Linear Regression w/ Closed Form Solution

    print("Q3:")
    Y = array[:, 1]
    inner = np.dot(X_normalized.T, X_normalized)
    inner_inverse = np.linalg.inv(inner)
    weights = (inner_inverse @ X_normalized.T) @ Y
    
    print(weights)
    
    #Question 4 - Linear Regression w/ Gradient Descent
    
    LEARNING_RATE = 0.6 #IMPORTANT: You will tune this so that the gradient descent converges
    
    gd_X = torch.tensor(X_normalized)
    gd_Y = torch.tensor(Y)
    gd_weights = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    n = len(gd_Y)
    losses = np.zeros(NUM_ITERS)
    
    print("Q4a:")
    for iter in range(NUM_ITERS):
        loss = (((gd_X @ gd_weights) - gd_Y) ** 2).sum() / n
        
        losses[iter] = loss.item()
        
        #Prints the weight and bias every 20 iterations
        if iter % 20 == 0:            
            print(gd_weights.detach().numpy())
        
        #Performs a backward pass through the computation graph
        #After this line, the gradient of the loss with respect to the weights is in gd_weights.grad
        loss.backward()

        #Performs one step of gradient descent
        with torch.no_grad():
            gd_weights -= LEARNING_RATE * gd_weights.grad

        #Resets the computation graph
        gd_weights.grad.zero_()

    plt.figure()
    plt.plot(range(NUM_ITERS), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.jpg")

    print("Q4b:", LEARNING_RATE)
    print("Q4c: I tried different weights starting with 0.01 and scaling by a factor of 10 until the rate started diverging. Then I tried increasingly smaller increments until I found a very close convergent learning rate")
    
    #Question 5 - Prediction
    
    y_hat = (weights[0] * ((2024 - min)/(max - min))) + weights[1]
    y_hat = y_hat[0]
    print("Q5: " + str(y_hat))


    #Question 6 - Model Interpretation
    symbol = None
    if weights[0] > 0:
        symbol = ">"
    elif weights[0] < 0:
        symbol = "<"
    else:
        symbol = "="

    print("Q6a: " + symbol)
    print("Q6b: The weight w is proportional to the expected change in the label from the last feature to the current. A positive w indicates increasing prediction value, and a negative w indicates a decreasing prediction value. If w is 0 then that feature has no effect on the prediction value.")


    #Question 7 - Model Limitations
    x_star = (((-weights[1]) * (max - min)) / (weights[0])) + min
    x_star = x_star[0]
    print("Q7a: " + str(x_star))
    print("Q7b: x* is a somewhat compelling prediction. It gives us a good idea of what may happen assuming future data follows the trend of past data. Unfortunately, this doesn't take into account compounding environmental factors(such as the ice-albedo affect) that have had a high impact that may decrease over time, or that have had little to no impact but may have a stronger impact over time. Also human intervention is not adjusted for, which is likely to drastically alter the outcome in some way.")
