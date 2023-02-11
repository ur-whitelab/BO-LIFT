import numpy as np


def expected_improvement(probs, values, best):
    """Expected improvement for the given discrete distribution"""
    ei = np.sum(np.maximum(values - best, 0) * probs)
    return ei


def probability_of_improvement(probs, values, best):
    """Probability of improvement for the given discrete distribution"""
    pi = np.sum(np.cast[float](values > best) * probs)
    return pi


def upper_confidence_bound(probs, values, best, _lambda):
    """Upper confidence bound for the given discrete distribution"""
    mu = np.sum(values * probs)
    sigma = np.sqrt(np.sum((values - mu) ** 2 * probs))
    return mu + _lambda * sigma


def greedy(probs, values, best):
    """Greedy selection (most likely point) for the given discrete distribution"""
    return values[np.argmax(probs)]
