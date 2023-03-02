import numpy as np
from scipy.stats import norm
from .llm_model import DiscreteDist, GaussDist


def expected_improvement(dist, best):
    """Expected improvement for the given discrete distribution"""
    if isinstance(dist, DiscreteDist):
        return expected_improvement_d(dist.probs, dist.values, best)
    elif isinstance(dist, GaussDist):
        return expected_improvement_g(dist.mean(), dist.std(), best)


def probability_of_improvement(dist, best):
    """Probability of improvement for the given discrete distribution"""
    if isinstance(dist, DiscreteDist):
        return probability_of_improvement_d(dist.probs, dist.values, best)
    elif isinstance(dist, GaussDist):
        return probability_of_improvement_g(dist.mean(), dist.std(), best)


def upper_confidence_bound(dist, best, _lambda):
    """Upper confidence bound for the given discrete distribution"""
    if isinstance(dist, DiscreteDist):
        return upper_confidence_bound_d(dist.probs, dist.values, best, _lambda)
    elif isinstance(dist, GaussDist):
        return upper_confidence_bound_g(dist.mean(), dist.std(), best, _lambda)


def greedy(dist, best):
    """Greedy selection (most likely point) for the given discrete distribution"""
    if isinstance(dist, DiscreteDist):
        return greedy_d(dist.probs, dist.values, best)
    elif isinstance(dist, GaussDist):
        return greedy_g(dist.mean(), dist.std(), best)


def expected_improvement_d(probs, values, best):
    """Expected improvement for the given discrete distribution"""
    ei = np.sum(np.maximum(values - best, 0) * probs)
    return ei


def probability_of_improvement_d(probs, values, best):
    """Probability of improvement for the given discrete distribution"""
    pi = np.sum(np.cast[float](values > best) * probs)
    return pi


def upper_confidence_bound_d(probs, values, best, _lambda):
    """Upper confidence bound for the given discrete distribution"""
    mu = np.sum(values * probs)
    sigma = np.sqrt(np.sum((values - mu) ** 2 * probs))
    return mu + _lambda * sigma


def greedy_d(probs, values, best):
    """Greedy selection (most likely point) for the given discrete distribution"""
    return values[np.argmax(probs)]


def expected_improvement_g(mean, std, best):
    """Expected improvement for the given Gaussian distribution"""
    z = (mean - best) / std
    ei = (mean - best) * norm.cdf(z) + std * norm.pdf(z)
    return ei


def probability_of_improvement_g(mean, std, best):
    """Probability of improvement for the given Gaussian distribution"""
    z = (mean - best) / std
    pi = norm.cdf(z)
    return pi


def upper_confidence_bound_g(mean, std, best, _lambda):
    """Upper confidence bound for the given Gaussian distribution"""
    return mean + _lambda * std


def greedy_g(mean, std, best):
    """Greedy selection (most likely point) for the given Gaussian distribution"""
    return mean
