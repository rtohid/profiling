#!/usr/bin/python3

__license__ = """ 
Copyright (c) 2020 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""

import argparse

profiler = argparse.ArgumentParser(
    description='Profiles input execution trees.')

profiler.add_argument(
    '-p',
    "--prefix",
    default="/tmp/__progiling__",
    help="where to store / dump the generated profiling info.")

args = profiler.parse_args()

print(f"profiler path: {args.prefix}")
