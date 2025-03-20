"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# 1. Multiplication
def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y

# 2. Identity function
def id(x: float) -> float:
    """Return the input unchanged."""
    return x

# 3. Addition
def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y

# 4. Negation
def neg(x: float) -> float:
    """Negate a number."""
    return -x

# 5. Less than
def lt(x: float, y: float) -> bool:
    """Check if x is less than y."""
    return x < y

# 6. Equality check
def eq(x: float, y: float) -> bool:
    """Check if two numbers are equal."""
    return x == y

# 7. Maximum
def max(x: float, y: float) -> float:
    """Return the larger of two numbers."""
    return x if x > y else y

# 8. Close comparison (within tolerance)
def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close in value within tolerance."""
    return abs(x - y) < 1e-2

# 9. Sigmoid function
def sigmoid(x: float) -> float:
    """Calculate the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

# 10. ReLU function (Rectified Linear Unit)
def relu(x: float) -> float:
    """Apply the ReLU activation function."""
    return max(0.0, x)

# 11. Natural logarithm
def log(x: float) -> float:
    """Calculate the natural logarithm."""
    return math.log(x)

# 12. Exponential function
def exp(x: float) -> float:
    """Calculate the exponential function."""
    return math.exp(x)

# 13. Reciprocal (1/x)
def inv(x: float) -> float:
    """Calculate the reciprocal."""
    return 1 / x if x != 0 else float('inf')  # Avoid division by zero

# 14. Derivative of log function with respect to x
def log_back(x: float, grad: float) -> float:
    """Compute the derivative of log(x) times a second argument."""
    return grad / x  # Derivative of log(x) is 1/x

# 15. Derivative of reciprocal function with respect to x
def inv_back(x: float, grad: float) -> float:
    """Compute the derivative of 1/x times a second argument."""
    return -grad / (x ** 2)  # Derivative of 1/x is -1/(x^2)

# 16. Derivative of ReLU function with respect to x
def relu_back(x: float, grad: float) -> float:
    """Compute the derivative of ReLU times a second argument."""
    return grad if x > 0 else 0  # Derivative of ReLU is 1 for x > 0, otherwise 0


# ## Task 0.3 - Higher-order Functions

# 1. map - Apply a function to each element in a list
def map(func: Callable[[float], float], iterable: Iterable[float]) -> list:
    """Apply a function to each element of an iterable."""
    return [func(x) for x in iterable]

# 2. zipWith - Apply a function to corresponding elements from two lists
def zipWith(func: Callable[[float, float], float], iterable1: Iterable[float], iterable2: Iterable[float]) -> list:
    """Apply a function to corresponding elements of two iterables."""
    return [func(x, y) for x, y in zip(iterable1, iterable2)]

# 3. reduce - Reduce an iterable to a single value by applying a function cumulatively
def reduce(func: Callable[[float, float], float], iterable: Iterable[float]) -> float:
    """Reduce an iterable to a single value using a function."""
    from functools import reduce
    return reduce(func, iterable)

# Implementing the required functions using the higher-order functions

# 4. negList - Negate all elements in a list using map
def negList(lst: list) -> list:
    """Negate all elements in a list."""
    return map(neg, lst)

# 5. addLists - Add corresponding elements from two lists using zipWith
def addLists(lst1: list, lst2: list) -> list:
    """Add corresponding elements from two lists."""
    return zipWith(add, lst1, lst2)

# 6. sum - Sum all elements in a list using reduce
def sum(lst: list) -> float:
    """Sum all elements in a list."""
    return reduce(add, lst)

# 7. prod - Calculate the product of all elements in a list using reduce
def prod(lst: list) -> float:
    """Calculate the product of all elements in a list."""
    return reduce(mul, lst)
