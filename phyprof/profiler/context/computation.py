__license__ = """ 
Copyright (c) 2020 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""

from typing import List, OrderedDict, Union
from phyprof.profiler.context.data import Cache


class Vector:
    def __init__(self, vector_length: int = 64, num_vectors: int = 4) -> None:
        self.vector_length = vector_length
        self.num_vectors = num_vectors


class CPU:
    def __init__(self, num_threads=1, num_sockets=1, vectorized=False):
        self.num_threads = num_threads
        self.num_sockets = num_sockets
        self.vectorized = vectorized


class GPU:
    def __init__(self, num_threads, num_gpus):
        self.num_threads = num_threads
        self.num_gpus = num_gpus
        self.sm = 16  # sm: streaming multiprocessor


class Node:
    num_nodes = 0

    def __init__(self, computation: List[Union[CPU, GPU]] = CPU(),
                 name: str=None) -> None:
        self.computation = computation


class Distributed:
    def __init__(self, *nodes):
        self.nodes = OrderedDict()
        self.add_node(nodes)

    def add_node(self, nodes):
        nodes_ = list()
        for node in nodes:
            nodes_.append(node)
        return nodes_
