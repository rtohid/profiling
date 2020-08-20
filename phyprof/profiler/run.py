from __future__ import absolute_import

__license__ = """ 
Copyright (c) 2020 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""

from phyprof.profiler.context.runtime import Distributed, CPU, GPU
from phyprof.profiler.context.runtime import Tiramisu


class Runtime:
    def __init__(self, runtime, *args, **kwargs):
        self.instance = runtime(*args, **kwargs)


class Run:
    def __init__(self, runtime=None, computation=None):
        self.runtime = Runtime(runtime)
        self.computation = computation
    
    def start(self):
        pass


node_0 = [GPU(1024, 1), CPU(16, 1)]
node_1 = [CPU(16, 4)]

computation = Distributed(node_0, node_1)
run = Run(HPX, computation)
print(run.runtime.instance)
print(run.computation)
