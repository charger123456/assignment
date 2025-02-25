"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y

def id(x: float) -> float:
    """Returns the input unchanged."""
    return x

def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y

def neg(x: float) -> float:
    """Negates a number."""
    return -x

def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another."""
    return x < y

def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal."""
    return x == y

def max(x: float, y: float) -> float:
    """Returns the larger of two numbers."""
    return x if x > y else y

def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    """Checks if two numbers are close in value."""
    return abs(x - y) < tol

def sigmoid(x: float) -> float:
    """Calculates the sigmoid function."""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)

def relu(x: float) -> float:
    """Applies the ReLU activation function."""
    return max(0, x)

def log(x: float) -> float:
    """Calculates the natural logarithm."""
    if x <= 0:
        raise ValueError("log input must be positive")
    return math.log(x)

def exp(x: float) -> float:
    """Calculates the exponential function."""
    return math.exp(x)

def inv(x: float) -> float:
    """Calculates the reciprocal."""
    if x == 0:
        raise ValueError("Cannot divide by zero")
    return 1 / x

def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second argument."""
    if x <= 0:
        raise ValueError("log_back input must be positive")
    return d / x

def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second argument."""
    if x == 0:
        raise ValueError("Cannot divide by zero")
    return -d / (x * x)

def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU times a second argument."""
    return d if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(fn: Callable[[float], float], iter: Iterable[float]) -> Iterable[float]:
    return [fn(x) for x in iter]

def zipWith(fn: Callable[[float, float], float], iter1: Iterable[float], iter2: Iterable[float]) -> Iterable[float]:
    return [fn(x, y) for x, y in zip(iter1, iter2)]

def reduce(fn: Callable[[float, float], float], iter: Iterable[float], start: float) -> float:
    result = start
    for x in iter:
        result = fn(result, x)
    return result

def negList(lst: Iterable[float]) -> Iterable[float]:
    return map(neg, lst)

def addLists(list1: Iterable[float], list2: Iterable[float]) -> Iterable[float]:
    return zipWith(add, list1, list2)

def sum(lst: Iterable[float]) -> float:
    return reduce(add, lst, 0)

def prod(lst: Iterable[float]) -> float:
    return reduce(mul, lst, 1)
