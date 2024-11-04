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


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function."""
    return max(0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal."""
    return 1 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg."""
    return y / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg."""
    return -y / (x ** 2)


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg."""
    return y if x > 0 else 0


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

def map(func: Callable[[float], float], iterable: Iterable[float]) -> Iterable[float]:
    """Applies a given function to each element of an iterable"""
    return [func(item) for item in iterable]


def zipWith(func: Callable[[float, float], float], iterable1: Iterable[float], iterable2: Iterable[float]) -> Iterable[
    float]:
    """Combines elements from two iterables using a given function."""
    return [func(a, b) for a, b in zip(iterable1, iterable2)]


def reduce(func: Callable[[float, float], float], iterable: Iterable[float], initial: float) -> float:
    """Reduces an iterable to a single float value using a given function."""
    result = initial
    for item in iterable:
        result = func(result, item)
    return result


def negList(iterable: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map"""
    return map(neg, iterable)


def addLists(iterable1: Iterable[float], iterable2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(add, iterable1, iterable2)


def sum(iterable: Iterable[float]) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(add, iterable, 0)


def prod(iterable: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce"""
    return reduce(mul, iterable, 1)
# TODO: Implement for Task 0.3.
