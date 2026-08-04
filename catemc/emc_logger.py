#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ase.ga import get_raw_score
from ase.parallel import parprint
import time


class EMCLogger(object):

    def __init__(self, verbose=True, acceptance_key='accepted'):

        self.verbose = verbose
        self.acceptance_key = acceptance_key
        self.start_time = time.perf_counter()

    def generation(self, generation, candidates, population):

        n_moves = len(candidates)
        n_accept = sum(a.info['key_value_pairs'][self.acceptance_key] for a in candidates)
        rate = 0. if n_moves == 0 else n_accept / n_moves
        best = max(get_raw_score(a) for a in population)
        elapsed = (time.perf_counter() - self.start_time) / 60.
        parprint('Generation {0}: acceptance {1}/{2} ({3:.3f}), best_raw_score {4:.8f}, elapsed {5:.1f} min'.format(
                generation, n_accept, n_moves, rate, best, elapsed))
