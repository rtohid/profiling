__license__ = """ 
Copyright (c) 2020 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""

from collections import defaultdict
from typing import DefaultDict


class HPX:
    def __init__(
        self,
        num_threads: int,
        num_localities: int,
        runtime_config: DefaultDict = defaultdict(lambda: None)
    ) -> None:

        self.num_threads = num_threads
        self.num_localities = num_localities

        self.runtime_config = runtime_config


class PhySL:
    def __init__(self, config=defaultdict(lambda: None)) -> None:
        self.config = config