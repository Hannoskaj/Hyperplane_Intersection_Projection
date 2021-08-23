#!/usr/bin/env python
# coding: utf-8
"""
Functions that did not fit in the classification.
"""


import numpy as N
import scipy as S
import scipy.linalg as SL
import scipy.stats as SS
import scipy.sparse as SP
import scipy.optimize as SO
import tables
import time
from pathlib import Path
import pandas
import collections


def max_entangled_state(d):
    id_state = N.eye(d) / d
    # We take tensor product of id states, and switch the axes
    prod_id_states = N.tensordot(id_state, id_state, axes=0)
    return prod_id_states.transpose([0, 2, 1, 3])


def Kirreg(p):
    Kraus1 = N.eye(p, dtype=complex)
    Kraus1[p - 1, p - 1] = 0
    Kraus2 = N.zeros((p, p), dtype=complex)
    Kraus2[0, p - 1] = 1
    return N.stack([Kraus1, Kraus2])
