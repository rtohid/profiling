__license__ = """ 
Copyright (c) 2020 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""


class Cache:
    def __init__(self, l1: None = int, l2: None = int, l3: None = int) -> None:
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        self.cache_line = None


class Memory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
