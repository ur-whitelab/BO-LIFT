"""utilities for building and selecting from a pool"""
from typing import List, Any, Callable
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from collections import OrderedDict
import re

class Pool:
    """Class for sampling from pool of possible data points

    Example:
        >>> pool = Pool(['a', 'b', 'c', 'd', 'e'])
        >>> pool.sample(3)
        ['a', 'd', 'c']
        >>> pool.choose('a')
        >>> pool.sample(3)
        ['b', 'c', 'd']
        >>> pool.approx_sample('a', 3)
        ['b', 'c', 'd']
    """

    def __init__(self, pool: List[Any], formatter: Callable = lambda x: str(x)) -> None:
        if type(pool) is not list:
            raise TypeError("Pool must be a list")
        self._pool = pool
        self._selected = []
        self._available = pool[:]
        self.format = formatter
        self._db = FAISS.from_texts(
            [formatter(x) for x in pool],
            OpenAIEmbeddings(),
            metadatas=[dict(data=p) for p in pool],
        )

    def sample(self, n: int) -> List[str]:
        """Sample n items from the pool"""
        if n > len(self._available):
            raise ValueError("Not enough items in pool")
        samples = np.random.choice(self._available, size=n, replace=False)
        return samples

    def choose(self, x: str) -> None:
        """Choose a specific item from the pool"""
        if x not in self._available:
            raise ValueError("Item not in pool")
        self._selected.append(x)
        self._available.remove(x)

    def approx_sample(self, x: str, k: int) -> None:
        """Given an approximation of x, return k similar"""
        # want to select extra, then remove previously chosen
        _k = k + len(self._selected)
        docs = self._db.max_marginal_relevance_search(x, k=_k, fetch_k=5 * _k)
        docs = [d.metadata["data"] for d in docs]
        # remove previously chosen
        docs = [d for d in docs if d not in self._selected]
        # select k
        return docs[:k]

    def reset(self) -> None:
        """Reset the pool"""
        self._selected = []
        self._available = self._pool[:]

    def __len__(self) -> int:
        return len(self._pool)

    def __repr__(self) -> str:
        return f"Pool of {len(self)} items with {len(self._selected)} selected"

    def __str__(self) -> str:
        return f"Pool of {len(self)} items with {len(self._selected)} selected"

    def __iter__(self):
        return iter(self._available)


class TreeNode:
    def __init__(self, name: str , value: Any, parent: 'TreeNode' = None):
        self.name = name
        self.value = value
        self._parent = parent
        self._children = []

    def add_child(self, node) -> None:
        self._children.append(node)
        node.set_parent(self)

    def set_parent(self, node) -> None:
        self._parent = node

    def get_parent(self) -> 'TreeNode':
        return self._parent

    def get_children_list(self) -> List['TreeNode']:
        return self._children
    
    def get_children_with_key(self, key):
        for child in self.get_children_list():
            if child.name == key:
                yield child
            else:
                yield from child.get_children_with_key(key)
    
    def get_branch(self):
        branch = OrderedDict({self.name: [self.value]})
        parent = self.get_parent()
        while parent.name != "root":
            branch[parent.name] = [parent.value]
            parent = parent.get_parent()
        return OrderedDict(reversed(branch.items()))

    def is_leaf(self):
        return len(self.get_children_list()) == 0

    def __str__(self):
        return f"< Node {self.name}: {self.value}>"
    
    def __repr__(self):
        return self.__str__()



class TreePool:
    '''
    
    Example:
        >>> pool = OrderedDict({'a': [1, 2, 3], 'b': [4, 5], 'c': [8]})
        >>> prompt_template = "A = {a}, B = {b}, C = {c}, A + B + C?"
        >>> pool = TreePool(pool, prompt_template)
        >>> pool._sample(3)
        [{'a':1, 'b':5, 'c':8}, {'a':2, 'b':4, 'c':8}, {'a':3, 'b':5, 'c':8},]
        >>> pool.sample(3)
        ["A = 2, B = 4, C = 8, A + B + C?", "A = 3, B = 5, C = 8, A + B + C?", "A = 1, B = 5, C = 8, A + B + C?",]
    '''

    def __init__(self, pool: OrderedDict[str, List[Any]], prompt_template:str, formatter: Callable = lambda x: str(x)) -> None:
        if type(pool) is not OrderedDict:
            raise TypeError("Pool must be a OrderedDict with variable names as keys and the range of possible values as values. Keys must be in order accordingly to the prompt")
        
        pattern = re.compile(r"\{(.*?)\}")
        if len(pattern.findall(prompt_template)) != len(pool):
            raise ValueError("Prompt template must have the same number of variables as the pool")
        
        self.prompt_template = prompt_template
        self._selected = []
        self._pool = pool
        self._root = TreeNode('root', None)
        self._build_tree()
        self._available = [leaf.get_branch() for leaf in self.get_leafs()]
        # Probably this self._available is not the best way to track all available paths. 
        # Refactor this. Looking over the tree may be more memory efficient (maybe slower?)
        self._selected = []


    def _build_tree(self):
        keys = list(self._pool.keys())
        for i, k in enumerate(keys):
            for child_v in self._pool[k]:
                if i == 0:
                    self._root.add_child(TreeNode(k, child_v))
                else:
                    for node in self.get_node_with_key(parent_key):
                        node.add_child(TreeNode(k, child_v))
            parent_key = k

    def get_node_with_key(self, key, root=None) -> List[TreeNode]:
        if root is None:
            root = self._root
        return [child for child in root.get_children_with_key(key)]
    
    def get_leafs(self, root=None) -> int:
        if root is None:
            root = self._root
        if root.is_leaf():
            return [root]
        else:
            return [leaf for child in root.get_children_list() for leaf in self.get_leafs(child)]

    def count_leafs(self, root=None) -> int:
        return len(self.get_leafs(root))

    def format_prompt(self, branch: OrderedDict[str, Any]) -> str:
        branch = {k: v[0] for k, v in branch.items()}
        return self.prompt_template.format(**branch)

    def partial_format_prompt(self, branch: OrderedDict[str, Any]) -> str:
        pattern = re.compile(r"\{(.*?)\}")
        matches = [match for match in re.finditer(pattern, self.prompt_template)]
        if len(branch) == len(matches):
            return self.format_prompt(branch)
        
        branch = {k: v[0] for k, v in branch.items()}
        partial_prompt_template = self.prompt_template[:matches[len(branch) - 1].end()]
        return partial_prompt_template.format(**branch)

    def sample(self, n) -> List[str]:
        samples = self._sample(n)
        return [self.format_prompt(sample) for sample in samples]

    def _sample(self, n) -> List[OrderedDict[str, Any]]:
        if n > self.count_leafs():
            raise ValueError("Not enough items in pool")
        samples = []
        while len(samples) < n:
            node = self._root
            while not node.is_leaf():
                node = np.random.choice(node.get_children_list())
            sample = node.get_branch()
            if sample not in samples:
                samples.append(sample)
        return samples
    

    def tree_sample(self, n, asktell) -> dict[str, Any]:
        if n > self.count_leafs():
            raise ValueError("Not enough items in pool")

        samples = []
        while len(samples) < n:
            node = self._root
            while not node.is_leaf():
                possible_x = [self.partial_format_prompt(child.get_branch()) for child in node.get_children_list()]
                asktell._ask(possible_x)
            sample = node.get_branch()
            if sample not in samples:
                samples.append(sample)
        return sample
    
    def choose(self) -> None:
        raise NotImplementedError("Choose is not implemented for TreePool")
    
    def approx_sample(self, x: str, k: int) -> None:
        raise NotImplementedError("Approximate sample is not implemented for TreePool. Consider using Pool instead")

    def reset(self) -> None:
        """Reset the pool"""
        self._root = TreeNode('root', None)
        self._build_tree()
        self._selected = []
        self._available = [leaf.get_branch() for leaf in self.get_leafs()]

    def __str__(self) -> str:
        return f"TreePool of {self.count_leafs()} with {len(self._selected)} selected"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __len__(self) -> int:
        return self.count_leafs()
    
    def _show_tree(self, root=None, level=0):
        """Print the tree
        DO NOT use this method for large trees. It is for debugging purposes only
        """
        if root is None:
            root = self._root
        print(f"{'|   '*level}{root}")
        for child in root.get_children_list():
            self._show_tree(child, level+1)
