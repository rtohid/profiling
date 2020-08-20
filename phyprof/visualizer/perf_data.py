#!/usr/bin/env python3

__license__ = """ 
Copyright (c) 2020 Shahrzad Shirzad (@scheherzade)
Copyright (c) 2020 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""

import datetime
import glob
from typing import Any, ChainMap, Mapping, Union

from phyprof.profiler.context.computation import CPU, Distributed, GPU

now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d-%H%M")
hpx_dir = '/home/shahrzad/repos/Blazemark/data/matrix/dmatdmatadd/01-04-2019-1027'


class Experiment:
    def __init__(self, target: Union[CPU, Distributed, GPU]) -> None:
        self.context = None
        self.runtime = None
        self.prefix = '/tmp/__profiling__'
        self.input = ChainMap()

    def config(self, input: Mapping, *args, **kwargs):
        pass


class Benchmark:
    def __init__(self, context: Experiment, **config) -> None:
        """
        :arg context: path to root directory of the benchmark.
        """
        self.input = input


hpx_dir = '/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/marvin/'
openmp_dir_2 = '/home/shahrzad/repos/Blazemark/data/openmp/04-27-2019/'
openmp_dir_1 = '/home/shahrzad/repos/Blazemark/data/openmp/all/'

perf_directory = '/home/shahrzad/repos/Blazemark/data/performance_plots/matrix/06-13-2019/marvin'


def create_dict(directory):
    thr = []
    repeats = []
    data_files = glob.glob(directory + '/*.dat')
    benchmark = ''
    benchmarks = []
    chunk_sizes = []
    block_sizes = []
    for filename in data_files:
        (repeat, benchmark, th, runtime, chunk_size, block_size_row,
         block_size_col) = filename.split('/')[-1].replace('.dat',
                                                           '').split('-')
        if benchmark not in benchmarks:
            benchmarks.append(benchmark)
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        if int(chunk_size) not in chunk_sizes:
            chunk_sizes.append(int(chunk_size))
        if block_size_row + '-' + block_size_col not in block_sizes:
            block_sizes.append(block_size_row + '-' + block_size_col)

    thr.sort()
    benchmarks.sort()
    repeats.sort()
    chunk_sizes.sort()
    block_sizes.sort()
    mat_sizes = {}

    d_all = {}
    d = {}
    for benchmark in benchmarks:
        d_all[benchmark] = {}
        d[benchmark] = {}
        for th in thr:
            d_all[benchmark][th] = {}
            d[benchmark][th] = {}
            for r in repeats:
                d_all[benchmark][th][r] = {}
                for bs in block_sizes:
                    d_all[benchmark][th][r][bs] = {}
                    d[benchmark][th][bs] = {}
                    for cs in chunk_sizes:
                        d_all[benchmark][th][r][bs][cs] = {}
                        d[benchmark][th][bs][cs] = {}

    data_files.sort()
    for filename in data_files:
        stop = False
        f = open(filename, 'r')

        result = f.readlines()[3:]
        benchmark = filename.split('/')[-1].split('-')[1]
        th = int(filename.split('/')[-1].split('-')[2])
        repeat = int(filename.split('/')[-1].split('-')[0])
        chunk_size = int(filename.split('/')[-1].split('-')[4])
        block_size = filename.split('/')[-1].split(
            '-')[5] + '-' + filename.split('/')[-1].split('-')[6][0:-4]
        size = []
        mflops = []
        for r in result:
            if "N=" in r:
                stop = True
            if not stop:
                size.append(int(r.strip().split(' ')[0]))
                mflops.append(float(r.strip().split(' ')[-1]))

        d_all[benchmark][th][repeat][block_size][chunk_size]['size'] = size
        d_all[benchmark][th][repeat][block_size][chunk_size]['mflops'] = mflops
        if 'size' not in d[benchmark][th][block_size][chunk_size].keys():
            d[benchmark][th][block_size][chunk_size]['size'] = size
            d[benchmark][th][block_size][chunk_size]['mflops'] = [0
                                                                  ] * len(size)
        if len(repeats) == 1 and repeat == 1:
            d[benchmark][th][block_size][chunk_size]['mflops'] = mflops
        elif len(repeats) > 1 and repeat != 1:
            d[benchmark][th][block_size][chunk_size]['mflops'] += mflops / (
                len(repeats) - 1)
        else:
            print("errrrrorrrrrrrrrrrr")
        if benchmark not in mat_sizes.keys():
            mat_sizes[benchmark] = size
    return (d, chunk_sizes, block_sizes, thr, benchmarks, mat_sizes)
