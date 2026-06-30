"""Calculator factory for NequIP worker pool mode.

This module is imported by each persistent worker to load the model ONCE.
The model path can be overridden via calculator_factory_kwargs["model_path"].
"""
import sys
import os
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)

DEFAULT_MODEL = "/fs/cephfs/data/qm_inorganics/RuMOx_OER/NequIP-OAM-S-stage6-h100-plain.nequip.pt2"


def create_calculators(model_path=None):
    from nequip.integrations.ase import NequIPCalculator
    path = model_path or DEFAULT_MODEL
    calc = NequIPCalculator.from_compiled_model(path, device="cuda")
    return [calc]
