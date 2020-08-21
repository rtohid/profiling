from __future__ import absolute_import

__license__ = """
Copyright (c) 2020 R. Tohid (@rtohid)
Copyright (c) 2020 Shahrzad Shirzad (@scheherzade)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""
from matplotlib import pyplot as plt
import subprocess
import time

from phyprof.profiler.context.computation import CPU, Distributed, GPU
from phyprof.profiler.context.runtime import HPX
from phyprof.profiler.run import Run

node_0 = [GPU(1024, 2), CPU(16, 1)]
node_1 = [CPU(16, 4)]

computation = Distributed(node_0, node_1)
run = Run(HPX, computation)

ts = {}
cores = [1, 2, 4] #, 8]
for n in cores:
    t = time.time()
    subprocess.call([
        "/home/stellar/src/hpx/build/bin/1d_stencil_4_parallel", "--nt=4", "--nx=5",
        f"--hpx:threads={n}"
    ])
    ts[n] = (time.time() - t)

i = 0
plt.figure(i)
plt.scatter(cores, [ts[n] for n in cores], marker='.')

plt.figure(i + 1)
plt.plot(cores, [ts[n] for n in cores], marker='.')
plt.xlabel("# cores")
plt.ylabel("Execution time(ns)")
plt.title("")

# optional
plt.grid(True)
plt.xscale('log')
plt.plot(cores, [ts[n] for n in cores], marker='.', label="run1")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# print(run.plot('cfg'))
# print(run.computation)