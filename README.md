# NeuralNetworkSimple

This is a small neural network with 5 neurons, 2 input, 2 hidden, and 1 output, that can predict the XOR function of 0 and 1. 

## How to Run

- Make sure you have python installed
- Save the script as NeuralNetworkSimple.py
- Run the script in your terminal with "python NeuralNetworkSimple.py"
- You can also copy and paste the code into an IDE or an online python compiler and run it there

## How it works 
It basically trains a neural network with 2 input neurons, 2 hidden nuerons, and 1 output neuron.
It prints the training every 200 epochs (One epoch is an entire runthrough of the data set) to show progress.
The data set would be:
- ([0, 0], 0),
- ([0, 1], 1),
- ([1, 0], 1),
- ([1, 1], 0)

An XOR function means that if the two values are the same, it should return 0, and if they are different, it should return 1.

## Resources
Neural networks explained - https://news.mit.edu/2017/explained-neural-networks-deep-learning-0414

