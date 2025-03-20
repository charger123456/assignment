import math
import random
import streamlit as st
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# Dataset Generation Functions
def make_pts(N):
    """Generate N random 2D points."""
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """Class to hold the dataset, including X (features) and y (labels)."""
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N):
    """
    Generate a simple classification dataset.
    The points are classified based on the value of x_1.
    If x_1 < 0.5, label is 1, else label is 0.

    Args:
    ----
        N: Number of points.

    Returns
    -------
        A Graph object containing the dataset.
    """
    X = make_pts(N)
    y = [1 if x_1 < 0.5 else 0 for x_1, x_2 in X]
    return Graph(N, X, y)


def diag(N):
    """
    Generate a dataset where the points are classified based on the sum of x_1 and x_2.
    If x_1 + x_2 < 0.5, label is 1, else label is 0.

    Args:
    ----
        N: Number of points.

    Returns
    -------
        A Graph object containing the dataset.
    """
    X = make_pts(N)
    y = [1 if x_1 + x_2 < 0.5 else 0 for x_1, x_2 in X]
    return Graph(N, X, y)


def split(N):
    """
    Generate a dataset where points are classified as 1 if x_1 is less than 0.2
    or greater than 0.8. Otherwise, the label is 0.

    Args:
    ----
        N: Number of points.

    Returns
    -------
        A Graph object containing the dataset.
    """
    X = make_pts(N)
    y = [1 if x_1 < 0.2 or x_1 > 0.8 else 0 for x_1, x_2 in X]
    return Graph(N, X, y)


def xor(N):
    """
    Generate a XOR dataset.
    The points are classified based on the XOR of x_1 and x_2.
    If x_1 < 0.5 and x_2 > 0.5, or x_1 > 0.5 and x_2 < 0.5, label is 1, else label is 0.

    Args:
    ----
        N: Number of points.

    Returns
    -------
        A Graph object containing the dataset.
    """
    X = make_pts(N)
    y = [1 if (x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5) else 0 for x_1, x_2 in X]
    return Graph(N, X, y)


def circle(N):
    """
    Generate a circular classification dataset.
    Points inside a circle of radius 0.1 are labeled 1, outside labeled 0.

    Args:
    ----
        N: Number of points.

    Returns
    -------
        A Graph object containing the dataset.
    """
    X = make_pts(N)
    y = [1 if (x_1 - 0.5)*2 + (x_2 - 0.5)*2 > 0.1 else 0 for x_1, x_2 in X]
    return Graph(N, X, y)


def spiral(N):
    """
    Generate a spiral classification dataset.
    The points form two spirals. One is labeled 0 and the other 1.

    Args:
    ----
        N: Number of points.

    Returns
    -------
        A Graph object containing the dataset.
    """
    def x(t):
        return t * math.cos(t) / 20.0

    def y(t):
        return t * math.sin(t) / 20.0

    X = [(x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5) for i in range(5 + 0, 5 + N // 2)]
    X = X + [(y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5) for i in range(5 + 0, 5 + N // 2)]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {'Simple': simple, 'Diag': diag, 'Split': split, 'Xor': xor, 'Circle': circle, 'Spiral': spiral}


# Streamlit Visualization
st.title('Interactive Dataset Visualizer')

# Sidebar for dataset selection
dataset_choice = st.sidebar.selectbox('Select a dataset:', list(datasets.keys()))
N = st.sidebar.slider('Number of points', 100, 1000, 300)

# Generate the selected dataset
graph = datasets[dataset_choice](N)

# Extract X and y values
X = graph.X
y = graph.y

# Plot the dataset
plt.figure(figsize=(8, 6))
plt.scatter([x_1 for x_1, _ in X], [x_2 for _, x_2 in X], c=y, cmap=plt.cm.Paired)
plt.title(f'{dataset_choice} Dataset')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Display the plot in Streamlit
st.pyplot(plt)

# Display the dataset parameters
st.write(f"Dataset: {dataset_choice}")
st.write(f"Total points: {N}")
st.write(f"Labels distribution: {sum(y)} points labeled 1, {N - sum(y)} points labeled 0")
