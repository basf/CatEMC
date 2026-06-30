"""Example calculator factory for CatEMC WorkerPool runs.

Each persistent worker imports this module once and calls create_calculators().
Edit this file for your own model/calculator stack; no CatEMC core changes are
needed as long as the function returns a list of ASE calculators.

Environment variables understood by this example:
    CATEMC_CALC_KIND     nequip | metatomic | mace | ase_emt
    CATEMC_MODEL_PATH    path to the MLIP model
    CATEMC_DEVICE        cuda | cpu, default cuda
    CATEMC_USE_REPULSIVE 1/0, prepend CatEMC RepulsivePotential if 1
"""
import os
import sys

try:
    sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
except Exception:
    pass


def create_calculators(model_path=None,
                       model_kind=None,
                       device=None,
                       use_repulsive=None,
                       metatomic_variant="react-r2scan"):
    calculators = []
    model_kind = (model_kind or os.environ.get("CATEMC_CALC_KIND", "metatomic")).lower()
    model_path = model_path or os.environ.get("CATEMC_MODEL_PATH")
    device = device or os.environ.get("CATEMC_DEVICE", "cuda")
    if use_repulsive is None:
        use_repulsive = os.environ.get("CATEMC_USE_REPULSIVE", "1") not in {"0", "false", "False"}

    if use_repulsive:
        from catemc.repulsive import RepulsivePotential
        calculators.append(RepulsivePotential())

    if model_kind == "nequip":
        if model_path is None:
            raise ValueError("CATEMC_MODEL_PATH or model_path is required for NequIP")
        from nequip.integrations.ase import NequIPCalculator
        calculators.append(NequIPCalculator.from_compiled_model(model_path, device=device))
    elif model_kind == "metatomic":
        if model_path is None:
            raise ValueError("CATEMC_MODEL_PATH or model_path is required for metatomic")
        from metatomic_ase import MetatomicCalculator
        calculators.append(MetatomicCalculator(
            model_path, device=device,
            variants={"energy": metatomic_variant},
            extensions_directory=None,
        ))
    elif model_kind == "mace":
        if model_path is None:
            raise ValueError("CATEMC_MODEL_PATH or model_path is required for MACE")
        from mace.calculators import MACECalculator
        calculators.append(MACECalculator(
            model_paths=model_path,
            dispersion=False,
            device=device,
            default_dtype=os.environ.get("CATEMC_MACE_DTYPE", "float64"),
        ))
    elif model_kind == "ase_emt":
        # Tiny smoke-test calculator; not for production chemistry.
        from ase.calculators.emt import EMT
        calculators.append(EMT())
    else:
        raise ValueError(f"Unknown CATEMC_CALC_KIND/model_kind: {model_kind!r}")

    return calculators
