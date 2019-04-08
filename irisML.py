# guided project using this youtube link https://www.youtube.com/watch?v=ZzWaow1Rvho&list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So
# test

import numpy

# defining the Neural Network function. Using measurements m1 and m2 and the weights w1 and w2 and bias b.
def NN(m1, m2, w1, w2, b):
    z = m1 * w1 + m2 * w2 + b
    return sigmoid(z)

# defining the sigmoid function, which is used in the NN function to return a number between 0 and 1.
def sigmoid(x):
    return 1/(1 + numpy.exp(-x))

# defining random weights and bias to start the model
w1 = numpy.random.randn()
w2 = numpy.random.randn()
b = numpy.random.randn()

# showing first prediction:
# Input is a red flower with petal lenght 3 and petal width 1.5.
# If output is close to 1, the prediction is 'red'
# If the output is close to 0, the prediction is 'blue'.
print(NN(3, 1.5, w1, w2, b))

# below a sample of the iris dataset [length, width, "blue" if 0 | "red" if 1)
data = [[3, 1.5, 1], [2, 1, 0], [4, 1.5, 1], [3, 1, 0], [3.5, .5, 1], [2, .5, 0], [5.5, 1, 1], [ 1, 1, 0]]

# creating rand_data variable to input random petal measurements
# first a random flower is selected from the list of lists 'data'
rand_data = data[numpy.random.randint(len(data))]
# then m1 is the first measurement (length)
m1 = rand_data[0]
# m2 is the second measurement (width)
m2 = rand_data[1]

# creating output variable for 'red' and 'blue' for easier interpretation.
prediction = NN(m1, m2, w1, w2, b)
# the prediction_text variable will use the rounded output of the prediction as an index number
# to turn prediction_text into either '["blue","red"][0]' or '["blue","red"][1]'.
prediction_text = ["blue", "red"][int(numpy.round(prediction))]
print("Prediction: " + prediction_text)

# the answer is given by printing the 0 or 1 (the third value in each flower list).
answer = "True answer: " + ["blue", "red"][rand_data[2]]
print(answer)

#Next steps: Add learning to the model to increase prediction accuracy.

