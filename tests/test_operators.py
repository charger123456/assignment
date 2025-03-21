from typing import Callable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import lists

from minitorch import MathTest
import minitorch
from minitorch.operators import (
    add,
    addLists,
    eq,
    id,
    inv,
    inv_back,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
    sigmoid,
)

from .strategies import assert_close, small_floats

# ## Task 0.1 Basic hypothesis tests.


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    """Check that the main operators all return the same value of the python version"""
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if abs(x) > 1e-5:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a: float) -> None:
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a: float, b: float) -> None:
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a: float) -> None:
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a: float) -> None:
    """Check that a - 1.0 is always less than a"""
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a: float) -> None:
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a: float) -> None:
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


# ## Task 0.2 - Property Testing

# Sigmoid function properties
@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a: float) -> None:
    assert 0.0 <= sigmoid(a) <= 1.0
    assert_close(1 - sigmoid(a), sigmoid(-a))
    assert_close(sigmoid(0), 0.5)
    if a < 0:
        assert sigmoid(a) < sigmoid(a + 1)

# Transitive property of less-than
@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    if lt(a, b) and lt(b, c):
        assert lt(a, c)

# Symmetry of multiplication
@pytest.mark.task0_2
def test_symmetric() -> None:
    x, y = 3.0, 5.0
    assert_close(mul(x, y), mul(y, x))

# Distributive property of multiplication over addition
@pytest.mark.task0_2
def test_distribute() -> None:
    z, x, y = 2.0, 3.0, 4.0
    assert_close(mul(z, add(x, y)), add(mul(z, x), mul(z, y)))

# Other mathematical properties (e.g., commutative property of addition)
@pytest.mark.task0_2
def test_other() -> None:
    x, y = 2.0, 3.0
    assert_close(add(x, y), add(y, x))


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a: float) -> None:
    """Check properties of the sigmoid function, specifically
    * It is always between 0.0 and 1.0.
    * one minus sigmoid is the same as sigmoid of the negative
    * It crosses 0 at 0.5
    * It is  strictly increasing.
    """
    # TODO: Implement for Task 0.2.
    raise NotImplementedError("Need to implement for Task 0.2")


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    """Test the transitive property of less-than (a < b and b < c implies a < c)"""
    # TODO: Implement for Task 0.2.
    raise NotImplementedError("Need to implement for Task 0.2")


@pytest.mark.task0_2
def test_symmetric() -> None:
    """Write a test that ensures that :func:`minitorch.operators.mul` is symmetric, i.e.
    gives the same value regardless of the order of its input.
    """
    # TODO: Implement for Task 0.2.
    raise NotImplementedError("Need to implement for Task 0.2")


@pytest.mark.task0_2
def test_distribute() -> None:
    r"""Write a test that ensures that your operators distribute, i.e.
    :math:`z \times (x + y) = z \times x + z \times y`
    """
    # TODO: Implement for Task 0.2.
    raise NotImplementedError("Need to implement for Task 0.2")


@pytest.mark.task0_2
def test_other() -> None:
    """Write a test that ensures some other property holds for your functions."""
    # TODO: Implement for Task 0.2.
    raise NotImplementedError("Need to implement for Task 0.2")


# ## Task 0.3  - Higher-order functions

# These tests check that your higher-order functions obey basic
# properties.


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a: float, b: float, c: float, d: float) -> None:
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    """Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls: List[float]) -> None:
    assert_close(sum(ls), minitorch.operators.sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x: float, y: float, z: float) -> None:
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls: List[float]) -> None:
    check = negList(ls)
    for i, j in zip(ls, check):
        assert_close(i, -j)


# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn: Tuple[str, Callable[[float], float]], t1: float) -> None:
    name, base_fn = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: Tuple[str, Callable[[float, float], float]], t1: float, t2: float
) -> None:
    name, base_fn = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a: float, b: float) -> None:
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
    from typing import List

import pytest

from hypothesis import given

from hypothesis.strategies import lists, floats

from minitorch.operators import (

map, zipWith, reduce, negList, addLists, sum, prod

)

# Test for map function

@pytest.mark.task0_3

@given(lists(floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100)))

def test_map(lst: List[float]) -> None:

"""Test that map correctly applies a function to all elements."""

squared = map(lambda x: x**2, lst)

assert all(a == b**2 for a, b in zip(squared, lst))

# Test for zipWith function

@pytest.mark.task0_3

@given(lists(floats(min_value=-100, max_value=100), min_size=3, max_size=3),

lists(floats(min_value=-100, max_value=100), min_size=3, max_size=3))

def test_zipWith(lst1: List[float], lst2: List[float]) -> None:

"""Test that zipWith correctly applies a function element-wise."""

summed = zipWith(lambda x, y: x + y, lst1, lst2)

assert all(a == b + c for a, b, c in zip(summed, lst1, lst2))

# Test for reduce function

@pytest.mark.task0_3

@given(lists(floats(min_value=-100, max_value=100), min_size=1))

def test_reduce(lst: List[float]) -> None:

"""Test that reduce correctly reduces a list."""

computed_sum = reduce(lambda x, y: x + y, lst)

expected_sum = sum(lst)

assert computed_sum == expected_sum

# Test for negList function

@pytest.mark.task0_3

@given(lists(floats(min_value=-100, max_value=100)))

def test_negList(lst: List[float]) -> None:

"""Test that negList correctly negates all elements."""

negated = negList(lst)

assert all(a == -b for a, b in zip(negated, lst))

# Test for addLists function

@pytest.mark.task0_3

@given(lists(floats(min_value=-100, max_value=100), min_size=3, max_size=3),

lists(floats(min_value=-100, max_value=100), min_size=3, max_size=3))

def test_addLists(lst1: List[float], lst2: List[float]) -> None:

"""Test that addLists correctly adds corresponding elements."""

added = addLists(lst1, lst2)

assert all(a == b + c for a, b, c in zip(added, lst1, lst2))

# Test for sum function

@pytest.mark.task0_3

@given(lists(floats(min_value=-100, max_value=100), min_size=1))

def test_sum(lst: List[float]) -> None:

"""Test that sum correctly sums the elements in a list."""

assert sum(lst) == reduce(lambda x, y: x + y, lst)

# Test for prod function

@pytest.mark.task0_3

@given(lists(floats(min_value=1, max_value=100), min_size=1))

def test_prod(lst: List[float]) -> None:

"""Test that prod correctly calculates the product of the elements in a list."""

assert prod(lst) == reduce(lambda x, y: x * y, lst)
