from __future__ import absolute_import

__license__ = """ 
Copyright (c) 2020 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""

from typing import Union

from phyprof.profiler.context.computation import Distributed, Node


class Runtime:
    def __init__(self, runtime, *args, **kwargs):
        self.instance = runtime(*args, **kwargs)


class Run:
    def __init__(self,
                 runtime: Runtime = None,
                 computation=Union[Distributed, Node]):
        self.runtime = runtime
        self.computation = computation

    def start(self):
        pass
