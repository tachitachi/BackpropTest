import random
import backprop as bp
from backprop import Number


def Linear2d(w, x):
    return bp.sigmoid(x[0] * w[0] + x[1] * w[1] + w[2])

if __name__ == '__main__':
    X = [ [1.2, 0.7], [-0.3, 0.5], [3, 2.5] ] # array of 2-dimensional data
    y = [1, -1, 1] # array of labels
    w = [0.1, 0.2, 0.3] # example: random numbers
    alpha = 0.1 # regularization strength
    margin = 1

    # Convert w to autograd variables
    w = list(map(lambda x: Number(x), w))

    opt = bp.SGD(w, lr=1e-2)

    for i in range(1000):
        idx = random.randint(0, len(X) - 1)

        d = X[idx]
        label = y[idx]

        output = Linear2d(w, d)

        # SVM margin loss
        #loss = bp.maximum(0, -label * output + margin) + alpha * (w[0] * w[0] + w[1] * w[1])

        # Binary Cross Entropy
        # -y log p
        loss = -label * bp.log(output)

        opt.zero_grad()
        loss.backward()
        opt.step()

    for d, label in zip(X, y):
        print(Linear2d(w, d), label)