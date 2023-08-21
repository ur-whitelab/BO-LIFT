import bolift
from bolift import TreePool, TreeNode
import numpy as np
import pytest
from collections import OrderedDict

np.random.seed(0)

@pytest.fixture
def prompt_template():
    return "A = {a}, B = {b}, C = {c}, A + B + C?"

def test_tree_node_get_children_with_key():
    root = TreeNode("root", None)
    child1 = TreeNode("a", 1)
    child2 = TreeNode("b", 2)
    child3 = TreeNode("a", 3)
    
    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(child3)

    children = list(root.get_children_with_key("a"))
    assert child1 in children
    children = list(child1.get_children_with_key("a"))
    assert child3 in children

def test_tree_node_get_branch():
    root = TreeNode("root", None)
    child1 = TreeNode("a", 1)
    child2 = TreeNode("b", 2)
    
    root.add_child(child1)
    child1.add_child(child2)

    assert child2.get_branch() == OrderedDict([("a", [1]), ("b", [2])])

def test_tree_pool_creation(prompt_template):
    pool = OrderedDict({'a': [1, 2, 3], 'b': [4, 5], 'c': [8]})
    tree_pool = TreePool(pool, prompt_template)
    assert str(tree_pool) == "TreePool of 6 with 0 selected"

def test_tree_pool_sample(prompt_template):
    pool = OrderedDict({'a': [1, 2, 3], 'b': [4, 5], 'c': [8]})
    tree_pool = TreePool(pool, prompt_template)
    samples = tree_pool.sample(3)
    assert len(samples) == 3
    assert any("A = 1" in s for s in samples)
    assert any("B = 5" in s for s in samples)
    assert any("C = 8" in s for s in samples)

@pytest.mark.parametrize("key", [('b',2), ('c',6)])
def test_tree_get_node_with_key(key, prompt_template):
    tree_data = OrderedDict([
        ("a", [1]),
        ("b", [2, 4]),
        ("c", [3, 4, 5])
    ])
    root = TreePool(tree_data, prompt_template)
    
    children = list(root.get_node_with_key(key[0]))
    assert len(children) == key[1]


def test_tree_get_branch(prompt_template):
    tree_data = OrderedDict([
        ("a", [1]),
        ("b", [2]),
        ("c", [3, 4])
    ])
    root = TreePool(tree_data, prompt_template)
    child2 = list(root.get_node_with_key("b"))[0]
    assert child2.get_branch() == OrderedDict([("a", [1]), ("b", [2])])
    child3 = list(root.get_node_with_key("c"))[0]
    assert child3.get_branch() == OrderedDict([("a", [1]), ("b", [2]), ("c", [3])])
    child3 = list(root.get_node_with_key("c"))[1]
    assert child3.get_branch() == OrderedDict([("a", [1]), ("b", [2]), ("c", [4])])