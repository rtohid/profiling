#!/usr/bin/env python

import sys
import json
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import os
import re

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('blazemark.pdf')

benchmarks = {}

# determine number of cores...
num_cores = 0
for dirname, dirnames, filenames in os.walk('data'):
    for filename in filenames:
        (benchmark, cores, runtime, chunk_size, multiplyer) = filename.replace('.dat','').split('-')
        num_cores = max(num_cores, int(cores))

for dirname, dirnames, filenames in os.walk('data'):
    for filename in filenames:
        (benchmark, cores, runtime) = filename.split('-')
        runtime = runtime.split('.')[0]
        idx = int(cores) - 1
        if runtime == 'openmp':
            runtime = 'OpenMP'
        if runtime == 'hpx':
            runtime = 'HPX'     
        if runtime == 'boost':
            runtime = 'BOOST' 
        if runtime == 'cpp':
            runtime = 'CPP' 
        #print(benchmark, cores, runtime)
        f = open(os.path.join(dirname, filename))
        benchmark_name = None
        benchmark_variants = []
        variant = ''
        for s in f:
            s = s[:-1]
            if s == '':
                continue

            # parse the benchmark name
            if re.match(' \S+', s):
                benchmark_name = s.strip()[:-1]
                if not benchmark_name in benchmarks:
                    benchmarks[benchmark_name] = {'variants' : {}}
                continue

            # check if a new variant started
            if re.match('   [^N]\S+', s):
                variant_split = s.strip().split(' ')
                variant = '%s - %s' % (runtime, ' '.join(variant_split[:-1]))
                if not 'metric' in benchmarks[benchmark_name]:
                    benchmarks[benchmark_name]['metric'] = variant_split[-1][1:-2]

                if not variant in benchmarks[benchmark_name]['variants']:
                    benchmarks[benchmark_name]['variants'][variant] = {}

                continue

            if re.match('     \d+', s):
                result = s.strip().split(' ')
                N = int(result[0])
                if not N in benchmarks[benchmark_name]['variants'][variant]:
                    benchmarks[benchmark_name]['variants'][variant][N] = np.zeros(num_cores)
                benchmarks[benchmark_name]['variants'][variant][N][idx] = float(result[-1])

                #results.append((result[0], result[-1]))

print(num_cores)
print(np.arange(num_cores) + 1)

#pprint(benchmarks)

lines = ['-', '-.', '--', ':']


for benchmark, variants in benchmarks.items():
    print('')
    print(benchmark)
    Ns=list(list(list(variants.items())[0][1].items())[0][1].keys())
    # scan data for different N's
#    Ns = sorted(variants['variants'].items()[1][1].keys())

    fig, ax = plt.subplots()
    idx = 0
    for variant, data in variants['variants'].items():
        print(variant)
        if 'HPX' in variant or 'OpenMP' in variant:
            jdx = 0
            for N in Ns:
                ax.plot(np.arange(num_cores) + 1, data[N], label='%s, N=%s' % (variant, N), color='C%s'%idx, linestyle=lines[jdx])
                jdx = (jdx + 1)
                if jdx == len(lines):
                    break
        idx = idx + 1
    ax.set_title('%s' % (benchmark))
    ax.set_xticks(np.arange(num_cores) + 1)
    ax.set_xlim(0, num_cores + 1)
    ax.set_xlabel('#Cores')
    ax.set_ylabel(variants['metric'])
#    ax.legend()
    ax.legend(bbox_to_anchor=(1.1, 1.05))


    plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')

plt.show()
pp.close()

