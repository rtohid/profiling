# Copyright (c) 2020 R. Tohid
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

all: cleanpy

cleanpy:
	find . -name .pytest_cache | xargs rm -rf 
	find . -name __pycache__ | xargs rm -rf 

.PHONY:
