import random
import backprop as bp
from backprop import Tensor
import data
import numpy as np
from tqdm import tqdm

def model(x, w1, b1, w2, b2):
    net = bp.relu(bp.matmul(x, w1) + b1)
    net = bp.softmax(bp.matmul(net, w2) + b2)
    return net

def Linear(inputs, outputs, bias=True):
    w = Tensor((np.random.random((inputs, outputs)) * 2 - 1) * np.sqrt(2 / (inputs + outputs)))
    if bias:
        b = Tensor(np.zeros((outputs,)))
        return w, b
    return w


def accuracy(y, output):
    correct = np.sum(np.argmax(y.value, -1) == np.argmax(output.value, -1))
    total = y.shape[0]
    return correct / total, correct, total

if __name__ == '__main__':
   
    train = data.MNIST('train', one_hot=True)
    test = data.MNIST('test', one_hot=True)
    train_loader = data.DataLoader(train, batch_size=32, shuffle=True, repeat=False)
    test_loader = data.DataLoader(train, batch_size=32, shuffle=False, repeat=False)

    w1, b1 = Linear(28*28, 256)
    w2, b2 = Linear(256, 10)

    opt = bp.SGD([w1, b1, w2, b2], lr=1e-3)

    epochs = 5
    for _ in tqdm(range(epochs)):
        for i, (x, y) in enumerate(train_loader):
            
            x = Tensor(x.reshape((-1, 28*28)))
            y = Tensor(y)

            output = model(x, w1, b1, w2, b2)

            # Binary Cross Entropy
            # -y log p
            loss = bp.reduce_sum(-y * bp.log(output))

            opt.zero_grad()
            loss.backward()
            opt.step()

    
    correct = 0
    total = 0
    for i, (x, y) in tqdm(enumerate(test_loader)):
        x = Tensor(x.reshape((-1, 28*28)))
        y = Tensor(y)
        output = model(x, w1, b1, w2, b2)

        acc, c, t = accuracy(y, output)
        correct += c
        total += t

    print('Final accuracy: {}'.format(correct / total))