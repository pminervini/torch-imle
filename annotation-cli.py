#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

from solvers.dijkstra import get_solver


def main(argv):
    neighbourhood_fn = "8-grid"
    solver = get_solver(neighbourhood_fn)



if __name__ == '__main__':
    main(sys.argv[1:])
