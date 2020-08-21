from __future__ import absolute_import

__license__ = """ 
Copyright (c) 2020 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""


class Runtime:
    def __init__(self, runtime, *args, **kwargs):
        # self.instance = runtime()  # (*args, **kwargs)
        self.runtime = runtime


class Run:
    def __init__(self, runtime=None, computation=None):
        self.runtime = Runtime(runtime)
        self.computation = computation

    def start(self):
        pass
