"""Calculator factory for worker pool mode.

This module is imported by each worker process to create a persistent
calculator. The model is loaded ONCE per worker at startup and reused
across all EMC generations — eliminating the per-generation subprocess overhead.

Adapt this for your model (NequIP, MACE, PET, etc.).
"""


def create_calculators():
    """Return a list of ASE calculators (one per model to share across atoms).

    For most use cases, return a single calculator — the worker pool will
    assign one calculator per worker automatically.
    """
    # === NequIP / Allegro ===
    # import sys, os
    # sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
    # from nequip.integrations.ase import NequIPCalculator
    # calc = NequIPCalculator.from_compiled_model("model.nequip.pt2", device="cuda")
    # return [calc]

    # === MACE ===
    from mace.calculators import MACECalculator
    calc = MACECalculator(
        model_paths="path/to/model.model",
        device="cuda",
        default_dtype="float64",
    )
    return [calc]

    # === PET (Metatomic) ===
    # from metatomic.torch.ase_calculator import MetatomicCalculator
    # calc = MetatomicCalculator("model.pt", extensions_directory=None, device="cuda")
    # return [calc]
