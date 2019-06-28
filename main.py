import math
from collections import defaultdict

class UnaryBackward:
    def __init__(self, a):
        self.a = a
    def update(self, a_grad):
        self.a.accumulate_gradient(a_grad)
    def __call__(self, gradient):
        raise NotImplementedError

class BinaryBackward:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def update(self, a_grad, b_grad):
        self.a.accumulate_gradient(a_grad)
        self.b.accumulate_gradient(b_grad)
    def __call__(self, gradient):
        raise NotImplementedError


class AddBackward(BinaryBackward):
    def __call__(self, gradient):
        da = gradient
        db = gradient
        self.update(da, db)
    def __str__(self):
        return '<AddBackward>'

def add(a, b):
    if type(b) == float or type(b) == int:
        b = Number(b)
    ret = Number(a.value + b.value)
    ret.grad_fn = AddBackward(a, b)
    ret.parent_nodes.append(a)
    ret.parent_nodes.append(b)
    return ret


class MulBackward(BinaryBackward):
    def __call__(self, gradient):
        da = gradient * self.b.value
        db = gradient * self.a.value
        self.update(da, db)
    def __str__(self):
        return '<MulBackward>'

def mul(a, b):
    if type(b) == float or type(b) == int:
        b = Number(b)
    ret = Number(a.value * b.value)
    ret.grad_fn = MulBackward(a, b)
    ret.parent_nodes.append(a)
    ret.parent_nodes.append(b)
    return ret


class DivBackward(BinaryBackward):
    def __call__(self, gradient):
        a = self.a.value
        b = 1 / self.b.value

        da = 1 * b * gradient
        db = (-1 * (b**2)) * a * gradient

        self.update(da, db)
    def __str__(self):
        return '<DivBackward>'

def div(a, b):
    if type(b) == float or type(b) == int:
        b = Number(b)
    ret = Number(a.value / b.value)
    ret.grad_fn = DivBackward(a, b)
    ret.parent_nodes.append(a)
    ret.parent_nodes.append(b)
    return ret

class SigmoidBackward(UnaryBackward):
    def __call__(self, gradient):
        value = 1 / (1 + math.exp(-self.a.value))
        da = (value * (1 - value)) * gradient
        self.update(da)
    def __str__(self):
        return '<SigmoidBackward>'

def sigmoid(a):
    value = 1 / (1 + math.exp(-a.value))
    ret = Number(value)
    ret.grad_fn = SigmoidBackward(a)
    ret.parent_nodes.append(a)
    return ret


class ReluBackward(UnaryBackward):
    def __call__(self, gradient):
        da = gradient if a.value > 0 else 0
        self.update(da)
    def __str__(self):
        return '<ReluBackward>'

def relu(a):
    ret = Number(a.value if a.value > 0 else 0)
    ret.grad_fn = ReluBackward(a)
    ret.parent_nodes.append(a)
    return ret


class Number(object):
    def __init__(self, value, grad=None):
        self.value = value
        self.grad = grad
        self.grad_fn = None
        self.parent_nodes = []
    def __repr__(self):
        return str(self)
    def __str__(self):
        return '<{},{},{}>'.format(self.value, self.grad if self.grad is not None else ' ', self.grad_fn if self.grad_fn is not None else ' ')

    def zero_grad(self):
        self.grad = None

    def accumulate_gradient(self, grad):
        if self.grad is None:
            self.grad = 0
        self.grad += grad

    def _backward(self):
        if self.grad_fn is not None:
            self.grad_fn(self.grad)

    # HACK: This delays backproping from this node until all other dependencies are done
    def backward(self, gradient=None):
        if gradient is None and self.grad is None:
            self.grad = 1
        else:
            self.grad = gradient

        for node in self.build_graph():
            node._backward()

    # Return a dictionary of child->parent edges
    def get_edges(self):
        forward_edges = defaultdict(set)
        backward_edges = defaultdict(set)
        for n in self.parent_nodes:
            forward_edges[n].add(self)
            backward_edges[self].add(n)

            # Merge forward and backward dicts
            new_forward, new_backward = n.get_edges()
            for n in new_forward:
                forward_edges[n] = forward_edges[n].union(new_forward[n])
            for n in new_backward:
                backward_edges[n] = backward_edges[n].union(new_backward[n])
        return forward_edges, backward_edges

    # Return the first edge found, any will work
    def _get_first_edge(self, edges):
        if len(edges) > 0:
            node = list(list(edges.keys()))[0]
            return node
        return None

    # Run DFS until we reach a leaf node
    def _dfs(self, edges):
        e = self._get_first_edge(edges)
        while e is not None and len(edges[e]) > 0:
            e = list(edges[e])[0]
        return e

    def build_graph(self):
        forward_edges, backward_edges = self.get_edges()
        forward_edges[self] = set()

        # Follow forward edges with DFS, and use backward edges to remove from forward edges
        sorted_list = []

        e = self._dfs(forward_edges)
        while e is not None:
            sorted_list.append(e)
            for b in backward_edges[e]:
                forward_edges[b].remove(e)

            del forward_edges[e]
            e = self._dfs(forward_edges)

        return sorted_list

    def __radd__(self, other):
        return self.__add__(other)
    def __add__(self, other):
        return add(self, other)

    def __rsub__(self, other):
        return add(Number(other), self * -1)
    def __sub__(self, other):
        return add(self, other * -1)

    def __rmul__(self, other):
        return self.__mul__(other)
    def __mul__(self, other):
        return mul(self, other)

    def __rtruediv__(self, other):
        return div(Number(other), self)
    def __truediv__(self, other):
        return div(self, other)


class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.value -= p.grad * self.lr


if __name__ == '__main__':

    a = Number(1)
    b = Number(2)
    c = Number(-3)
    x = Number(-1)
    y = Number(3)

    opt = SGD([x, y])

    for i in range(100):

        output = sigmoid(a * x + b * y + c)

        print(output)

        opt.zero_grad()
        output.backward()
        opt.step()