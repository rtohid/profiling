from __future__ import absolute_import

__license__ = """ 
Copyright (c) 2020 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""

from phyprof.profiler.context.computation import CPU, GPU, Node, Distributed
from phyprof.profiler.context.runtime import PhySL
from phyprof.profiler.run import Run

node_0 = [GPU(1024, 2), CPU(16, 1)]
node_1 = [CPU(16, 4)]

computation = Distributed(node_0, node_1)
run = Run(PhySL, computation)
print(run.runtime.instance)
print(run.computation)