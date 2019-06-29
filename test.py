
import unittest
import random

import backprop as bp


class TestGradients(unittest.TestCase):
    epsilon = 1e-5
    def test_add2(self):
        def add2(a, b):
            return a + b
        for i in range(100):
            a = random.random() * 10 - 5
            b = random.random() * 10 - 5

            diffs = bp.compare_gradients(add2, [a, b])
            self.assertEqual(len(list(filter(lambda x: x > self.epsilon, diffs))), 0)

    def test_add3(self):
        def add3(a, b, c):
            return a + b + c
        for i in range(100):
            a = random.random() * 10 - 5
            b = random.random() * 10 - 5
            c = random.random() * 10 - 5

            diffs = bp.compare_gradients(add3, [a, b, c])
            self.assertEqual(len(list(filter(lambda x: x > self.epsilon, diffs))), 0)

    def test_add_repeat(self):
        def add3(a, b, c):
            return a + b + c + a + a + a
        for i in range(100):
            a = random.random() * 10 - 5
            b = random.random() * 10 - 5
            c = random.random() * 10 - 5

            diffs = bp.compare_gradients(add3, [a, b, c])
            self.assertEqual(len(list(filter(lambda x: x > self.epsilon, diffs))), 0)

    def test_mul2(self):
        def mul2(a, b):
            return a * b
        for i in range(100):
            a = random.random() * 10 - 5
            b = random.random() * 10 - 5

            diffs = bp.compare_gradients(mul2, [a, b])
            self.assertEqual(len(list(filter(lambda x: x > self.epsilon, diffs))), 0)

    def test_mul3(self):
        def mul3(a, b, c):
            return a * b * c
        for i in range(100):
            a = random.random() * 10 - 5
            b = random.random() * 10 - 5
            c = random.random() * 10 - 5

            diffs = bp.compare_gradients(mul3, [a, b, c])
            self.assertEqual(len(list(filter(lambda x: x > self.epsilon, diffs))), 0)

    def test_mul_repeat(self):
        def mul3(a, b, c):
            return a * b * c * a * a * a
        for i in range(100):
            a = random.random() * 10 - 5
            b = random.random() * 10 - 5
            c = random.random() * 10 - 5

            diffs = bp.compare_gradients(mul3, [a, b, c])
            self.assertEqual(len(list(filter(lambda x: x > self.epsilon, diffs))), 0)

if __name__ == '__main__':
    unittest.main()
