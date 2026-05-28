"""Calculator factory for NequIP worker pool mode.

This module is imported by each persistent worker to load the model ONCE.
"""
import sys
import os
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)

COMPILED_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "NequIP-OAM-S-stage6-h100-plain.nequip.pt2")

def create_calculators():
    from nequip.integrations.ase import NequIPCalculator
    calc = NequIPCalculator.from_compiled_model(COMPILED_MODEL, device="cuda")
    return [calc]
