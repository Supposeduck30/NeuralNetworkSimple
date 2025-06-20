import math
import random
import time

random.seed(time.time())

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

# XOR training data
data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

# Initialize weights for a 2-2-1 network
w1 = [random.uniform(-1, 1), random.uniform(-1, 1)]  # input to hidden neuron 1
w2 = [random.uniform(-1, 1), random.uniform(-1, 1)]  # input to hidden neuron 2
b1 = random.uniform(-1, 1)
b2 = random.uniform(-1, 1)
v = [random.uniform(-1, 1), random.uniform(-1, 1)]   # hidden to output
bo = random.uniform(-1, 1)

lr = 0.1
epochs = 2000

for epoch in range(epochs):
    total_loss = 0
    for (x1, x2), target in data:
        # Forward pass
        h1 = sigmoid(w1[0]*x1 + w1[1]*x2 + b1)
        h2 = sigmoid(w2[0]*x1 + w2[1]*x2 + b2)
        out = sigmoid(v[0]*h1 + v[1]*h2 + bo)
        error = out - target
        total_loss += error ** 2

        # Backpropagation
        d_out = 2 * error * sigmoid_derivative(v[0]*h1 + v[1]*h2 + bo)
        dv0 = d_out * h1
        dv1 = d_out * h2
        dbo = d_out

        dh1 = d_out * v[0] * sigmoid_derivative(w1[0]*x1 + w1[1]*x2 + b1)
        dh2 = d_out * v[1] * sigmoid_derivative(w2[0]*x1 + w2[1]*x2 + b2)

        dw1_0 = dh1 * x1
        dw1_1 = dh1 * x2
        db1 = dh1

        dw2_0 = dh2 * x1
        dw2_1 = dh2 * x2
        db2 = dh2

        # Update weights
        v[0] -= lr * dv0
        v[1] -= lr * dv1
        bo  -= lr * dbo

        w1[0] -= lr * dw1_0
        w1[1] -= lr * dw1_1
        b1   -= lr * db1

        w2[0] -= lr * dw2_0
        w2[1] -= lr * dw2_1
        b2   -= lr * db2

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, loss: {total_loss:.4f}")

print("\nFinal predictions:")
for (x1, x2), target in data:
    h1 = sigmoid(w1[0]*x1 + w1[1]*x2 + b1)
    h2 = sigmoid(w2[0]*x1 + w2[1]*x2 + b2)
    out = sigmoid(v[0]*h1 + v[1]*h2 + bo)
    print(f"Input: [{x1}, {x2}] → Predicted: {out:.4f}, Actual: {target}")
