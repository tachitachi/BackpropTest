import numpy as np
from collections import defaultdict

class UnaryBackward:
    def __init__(self, a, axis=None, keepdims=False):
        self.a = a
        self.axis = axis
        self.keepdims = keepdims
    def update(self, a_grad, a_shape=None):
        self.a.accumulate_gradient(a_grad, a_shape)
    def __call__(self, gradient):
        raise NotImplementedError

class BinaryBackward:
    def __init__(self, a, b, axis=None, keepdims=False):
        self.a = a
        self.b = b
        self.axis = axis
        self.keepdims = keepdims
    def update(self, a_grad, b_grad, a_shape=None, b_shape=None):
        self.a.accumulate_gradient(a_grad, a_shape)
        self.b.accumulate_gradient(b_grad, b_shape)
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
        b = Tensor(b)
    ret = Tensor(a.value + b.value)
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
        b = Tensor(b)
    ret = Tensor(a.value * b.value)
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
        b = Tensor(b)
    ret = Tensor(a.value / b.value)
    ret.grad_fn = DivBackward(a, b)
    ret.parent_nodes.append(a)
    ret.parent_nodes.append(b)
    return ret

class MatmulBackward(BinaryBackward):
    def __call__(self, gradient):
        da = np.dot(gradient, self.b.value.T)
        db = np.dot(self.a.value.T, gradient)
        self.update(da, db)
    def __str__(self):
        return '<MatmulBackward>'

def matmul(a, b):
    ret = Tensor(np.matmul(a.value, b.value))
    ret.grad_fn = MatmulBackward(a, b)
    ret.parent_nodes.append(a)
    ret.parent_nodes.append(b)
    return ret


class SigmoidBackward(UnaryBackward):
    def __call__(self, gradient):
        value = 1 / (1 + np.exp(-self.a.value))
        da = (value * (1 - value)) * gradient
        self.update(da)
    def __str__(self):
        return '<SigmoidBackward>'

def sigmoid(a):
    value = 1 / (1 + np.exp(-a.value))
    ret = Tensor(value)
    ret.grad_fn = SigmoidBackward(a)
    ret.parent_nodes.append(a)
    return ret


class ReluBackward(UnaryBackward):
    def __call__(self, gradient):
        da = gradient * (self.a.value > 0).astype(np.float32)
        self.update(da)
    def __str__(self):
        return '<ReluBackward>'

def relu(a):
    ret = Tensor(np.maximum(0, a.value))
    ret.grad_fn = ReluBackward(a)
    ret.parent_nodes.append(a)
    return ret


class MaxBackward(BinaryBackward):
    def __call__(self, gradient):
        is_max = (self.a.value >= self.b.value).astype(np.float32)
        da = gradient * is_max
        db = gradient * (1 - is_max)
        self.update(da, db)
    def __str__(self):
        return '<MaxBackward>'

def maximum(a, b):
    if not isinstance(a, Tensor):
        a = Tensor(a)
    if not isinstance(b, Tensor):
        b = Tensor(b)
    ret = Tensor(np.maximum(a.value, b.value))
    ret.grad_fn = MaxBackward(a, b)
    ret.parent_nodes.append(a)
    ret.parent_nodes.append(b)
    return ret


class LogBackward(UnaryBackward):
    def __call__(self, gradient):
        da = gradient * (1 / self.a.value)
        self.update(da)
    def __str__(self):
        return '<LogBackward>'

def log(a):
    ret = Tensor(np.log(a.value))
    ret.grad_fn = LogBackward(a)
    ret.parent_nodes.append(a)
    return ret

class ExpBackward(UnaryBackward):
    def __call__(self, gradient):
        da = gradient * np.exp(self.a.value)
        self.update(da)
    def __str__(self):
        return '<ExpBackward>'

def exp(a):
    ret = Tensor(np.exp(a.value))
    ret.grad_fn = ExpBackward(a)
    ret.parent_nodes.append(a)
    return ret


# reduce functions
class ReduceSumBackward(UnaryBackward):
    def __call__(self, gradient):
        da = gradient * 1
        # Need to add reduced dimensions back in
        # Does the order matter?
        if self.axis is not None and not self.keepdims:
            if type(self.axis) == int:
                da = np.expand_dims(da, self.axis)
            else:
                for ax in sorted(self.axis):
                    da = np.expand_dims(da, ax)
        #print('da', da.shape)
        self.update(da, da.shape)
    def __str__(self):
        return '<ReduceSumBackward>'


def reduce_sum(a, axis=None, keepdims=False):
    ret = Tensor(np.sum(a.value, axis=axis, keepdims=keepdims))
    ret.grad_fn = ReduceSumBackward(a, axis, keepdims=keepdims)
    ret.parent_nodes.append(a)
    return ret

class ReduceMeanBackward(UnaryBackward):
    def __call__(self, gradient):
        # Gradient is (1 / num_elements)
        axes = self.axis
        if type(axes) == int:
            axes = (axes,)

        if axes is None:
            num_elements = np.prod(self.a.shape)
        else:
            num_elements = np.prod([self.a.shape[ax] for ax in axes])

        da = gradient / num_elements
        # Need to add reduced dimensions back in
        # Does the order matter?
        if self.axis is not None and not self.keepdims:
            if type(self.axis) == int:
                da = np.expand_dims(da, self.axis)
            else:
                for ax in sorted(self.axis):
                    da = np.expand_dims(da, ax)
        self.update(da, da.shape)
    def __str__(self):
        return '<ReduceMeanBackward>'


def reduce_mean(a, axis=None, keepdims=False):
    ret = Tensor(np.mean(a.value, axis=axis, keepdims=keepdims))
    ret.grad_fn = ReduceMeanBackward(a, axis, keepdims=keepdims)
    ret.parent_nodes.append(a)
    return ret

class ReduceMaxBackward(UnaryBackward):
    def __call__(self, gradient):
        # Gradient is 1 where the value is max, and 0 otherwise
        da = gradient
        # Need to add reduced dimensions back in
        # Does the order matter?
        if self.axis is not None and not self.keepdims:
            if type(self.axis) == int:
                da = np.expand_dims(da, self.axis)
            else:
                for ax in sorted(self.axis):
                    da = np.expand_dims(da, ax)

        da = da * (self.a.value == np.max(self.a.value, self.axis, keepdims=True)).astype(np.float32)

        self.update(da, da.shape)
    def __str__(self):
        return '<ReduceMaxBackward>'

def reduce_max(a, axis=None, keepdims=False):
    ret = Tensor(np.max(a.value, axis=axis, keepdims=keepdims))
    ret.grad_fn = ReduceMaxBackward(a, axis, keepdims=keepdims)
    ret.parent_nodes.append(a)
    return ret

def softmax(a, axis=None):
    return _softmax(a - reduce_max(a, axis), axis)

def _softmax(a, axis=None):
    if axis is None:
        axis = len(a.shape) - 1
    return exp(a) / reduce_sum(exp(a), axis=axis, keepdims=True)

# Broadcasting rules:
# scalar - sum over everything
# trailing dimensions must either match, or be 1
#   if they don't match, sum over that dimension
# Prepend 1s to the smaller dimension

# TODO: Verify
# self.shape should always be equal to, or smaller than grad.shape?
# FALSE, if coming from a reduction function
def reduce_grads(grad, param_shape):
    # Compare trailing dimensions, sum over dim if mismatch
    sum_dims = list(map(lambda x: x[0], filter(lambda x: x[1] != x[2], zip(np.arange(len(grad.shape))[-len(param_shape):], grad.shape[-len(param_shape):], param_shape))))

    # Preprend 1s, which will automatically be summed over
    sum_dims = tuple(np.arange(len(grad.shape) - len(param_shape)).tolist() + sum_dims)

    # Sum gradients over broadcast dimensions, and reshape to target shape
    return grad.sum(axis=sum_dims).reshape(param_shape)


class Tensor(object):
    def __init__(self, value, dtype=np.float32, grad=None):
        if type(value) == int or type(value) == float:
            value = np.array(value, dtype=dtype)
        self.value = value
        self.dtype = dtype
        self.grad = grad
        self.grad_fn = None
        self.parent_nodes = []
    def __repr__(self):
        return str(self)
    def __str__(self):
        return '<{},{},{}>'.format(self.value, self.grad if self.grad is not None else ' ', self.grad_fn if self.grad_fn is not None else ' ')

    @property
    def shape(self):
        return self.value.shape
    
    def zero_grad(self):
        self.grad = None

    def accumulate_gradient(self, grad, shape=None):
        if self.grad is None:
            self.grad = np.zeros_like(self.value, dtype=self.dtype)
        if shape is None:
            shape = self.value.shape

        grads = reduce_grads(grad, shape)
        self.grad += grads

    def _backward(self):
        if self.grad_fn is not None:
            self.grad_fn(self.grad)

    def backward(self, gradient=None):
        if gradient is None and self.grad is None:
            self.grad = np.ones_like(self.value, dtype=self.dtype)
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
        return add(Tensor(other), self * -1)
    def __sub__(self, other):
        return add(self, other * -1)

    def __rmul__(self, other):
        return self.__mul__(other)
    def __mul__(self, other):
        return mul(self, other)

    def __rtruediv__(self, other):
        return div(Tensor(other), self)
    def __truediv__(self, other):
        return div(self, other)

    def __neg__(self):
        return mul(self, -1)



def numerical_gradients(func, params, h=1e-8):
    grads = []
    for idx in range(len(params)):
        p = list(map(lambda x: Tensor(x) if type(x) != Tensor else x, params))
        # f(x + h) - f(x) / h
        f_x = func(*p).value
        p[idx].value += h
        f_x_h = func(*p).value

        grads.append(reduce_grads((f_x_h - f_x) / h, p[idx].shape))
    return grads


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
    np.random.seed(1)
    x = Tensor(np.random.random((3, 5)))
    y = Tensor(np.random.random((3, 5, 2, 4)))
    w = Tensor(np.random.random((5, 2)))
    s = Tensor(np.random.random((5,)))

    a = Tensor(np.array([1, 2, 3, 4, 5, 6]).reshape((2, 3)))

    def f(x, y):
        return matmul(x, y)

    params = [x, w]
    z = f(*params)
    z.backward()

    grads = numerical_gradients(f, params)
    print(grads)