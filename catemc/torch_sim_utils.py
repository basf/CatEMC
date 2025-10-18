# torch_sim_fixatoms/wrappers.py
from typing import Sequence, Optional, Callable, Any
import torch


class FixAtoms:
    """
    Minimal FixAtoms helper.

    - fix_indices: sequence of integer atom indices to fix (0-based indices into the atomic positions array)
    - n_atoms: optional total atom count
    - device: optional torch.device for masks

    Notes:
    - This class provides helpers to mask forces, velocities and per-atom virials.
    - For large systems you can set `n_atoms` ahead of time and the device to avoid inferring from tensors.
    """

    def __init__(self, fix_indices: Sequence[int], n_atoms: Optional[int] = None, device: Optional[torch.device] = None):
        self.fix_indices = torch.as_tensor(list(fix_indices), dtype=torch.long)
        self._n_atoms = int(n_atoms) if n_atoms is not None else None
        self._device = device

    def set_n_atoms(self, n: int):
        self._n_atoms = int(n)

    def _ensure_device_and_n(self, tensor: Optional[torch.Tensor] = None):
        dev = self._device
        n = self._n_atoms
        if tensor is not None:
            dev = tensor.device
            if n is None:
                n = tensor.shape[0]
        return dev, n

    def forces_mask(self, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        dev, n = self._ensure_device_and_n(positions)
        if n is None:
            raise ValueError("Number of atoms unknown: provide n_atoms to FixAtoms or pass positions")
        mask = torch.ones((n, 3), dtype=torch.get_default_dtype(), device=dev)
        if self.fix_indices.numel() > 0:
            mask[self.fix_indices.to(dev)] = 0.0
        return mask

    def logical_mask(self, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        dev, n = self._ensure_device_and_n(positions)
        if n is None:
            raise ValueError("Number of atoms unknown: provide n_atoms to FixAtoms or pass positions")
        mask = torch.zeros((n,), dtype=torch.bool, device=dev)
        if self.fix_indices.numel() > 0:
            mask[self.fix_indices.to(dev)] = True
        return mask

    def apply_to_forces(self, forces: torch.Tensor) -> torch.Tensor:
        """
        Zero-out force rows corresponding to fixed atoms.
        Works for shape (N,3) or (B,N,3). This version is safe: it ignores any fixed indices
        that are >= N (so it won't raise indexing errors for padded/batched states).
        """
        if forces is None:
            return forces
        dev = forces.device
        idx = self.fix_indices.to(dev)
        out = forces.clone()
        # support (N,3) or (B,N,3)
        if out.dim() == 2:
            N = out.shape[0]
            valid = idx[idx < N]
            if valid.numel() > 0:
                out[valid] = 0.0
        elif out.dim() == 3:
            N = out.shape[1]
            valid = idx[idx < N]
            if valid.numel() > 0:
                out[:, valid, :] = 0.0
        else:
            raise ValueError(f"Unsupported forces tensor shape: {out.shape}")
        return out

    def zero_velocities(self, velocities: torch.Tensor) -> torch.Tensor:
        """
        Zero velocities for fixed atoms in a safe way (supports (N,3) and (B,N,3)).
        Ignores indices outside the current N dimension.
        """
        if velocities is None:
            return velocities
        dev = velocities.device
        idx = self.fix_indices.to(dev)
        out = velocities.clone()
        if out.dim() == 2:
            N = out.shape[0]
            valid = idx[idx < N]
            if valid.numel() > 0:
                out[valid] = 0.0
        elif out.dim() == 3:
            N = out.shape[1]
            valid = idx[idx < N]
            if valid.numel() > 0:
                out[:, valid, :] = 0.0
        else:
            raise ValueError(f"Unsupported velocities tensor shape: {out.shape}")
        return out

    def apply_to_virials(self, virials: torch.Tensor) -> torch.Tensor:
        """
        Zero per-atom virial contributions for fixed atoms in a safe way (ignores out-of-bounds indices).
        Accepts common per-atom virial shapes: (N,3,3), (B,N,3,3), (N,6), (B,N,6).
        """
        if virials is None:
            return virials
        dev = virials.device
        idx = self.fix_indices.to(dev)
        out = virials.clone()
        # (N,3,3)
        if out.dim() == 3 and out.shape[1:] == (3, 3):
            N = out.shape[0]
            valid = idx[idx < N]
            if valid.numel() > 0:
                out[valid, :, :] = 0.0
        # (B,N,3,3)
        elif out.dim() == 4 and out.shape[2:] == (3, 3):
            N = out.shape[1]
            valid = idx[idx < N]
            if valid.numel() > 0:
                out[:, valid, :, :] = 0.0
        # (N,6)
        elif out.dim() == 2 and out.shape[1] == 6:
            N = out.shape[0]
            valid = idx[idx < N]
            if valid.numel() > 0:
                out[valid, :] = 0.0
        # (B,N,6)
        elif out.dim() == 3 and out.shape[2] == 6:
            N = out.shape[1]
            valid = idx[idx < N]
            if valid.numel() > 0:
                out[:, valid, :] = 0.0
        else:
            raise ValueError(f"Unsupported virials tensor shape: {out.shape}")
        return out

    def restore_positions(self, positions: torch.Tensor, previous_positions: torch.Tensor) -> torch.Tensor:
        """
        Restore fixed atoms positions.

        This version supports:
          - previous_positions as full positions (N,3) or (B,N,3) -> replace whole arrays
          - previous_positions as *only* fixed-atom positions:
                (num_fixed,3)  -> placed into positions[idx]
                (B,num_fixed,3) -> placed into positions[:, idx, :]

        It safely ignores any fixed indices >= N (i.e. out-of-bounds indices won't be accessed).
        """
        out = positions.clone()
        dev = out.device
        idx = self.fix_indices.to(dev)

        # If previous_positions equals full positions shape, just replace
        if previous_positions is None:
            return out

        # previous_positions is full (N,3)
        if previous_positions.dim() == 2 and previous_positions.shape == out.shape:
            return previous_positions.to(dev)

        # previous_positions is full batched (B,N,3)
        if previous_positions.dim() == 3 and previous_positions.shape == out.shape:
            return previous_positions.to(dev)

        # previous_positions is per-fixed (num_fixed,3)
        if previous_positions.dim() == 2 and previous_positions.shape[0] == idx.numel():
            # positions could be (N,3) or (B,N,3)
            if out.dim() == 2:
                N = out.shape[0]
                valid = idx[idx < N]
                if valid.numel() > 0:
                    out[valid] = previous_positions.to(dev)[valid]
                return out
            elif out.dim() == 3:
                N = out.shape[1]
                valid = idx[idx < N]
                if valid.numel() > 0:
                    out[:, valid, :] = previous_positions.to(dev)[valid].unsqueeze(0)
                return out

        # previous_positions is batched-per-fixed (B, num_fixed, 3)
        if previous_positions.dim() == 3 and previous_positions.shape[1] == idx.numel():
            N = out.shape[1]
            valid = idx[idx < N]
            if valid.numel() > 0:
                # apply only the valid subset
                out[:, valid, :] = previous_positions.to(dev)[:, valid, :]
            return out

        # fallback: attempt to place by matching index counts (best-effort)
        # If nothing matched, return positions unchanged (safe fallback).
        return out


class ModelProxy:
    """
    Proxy that makes a model callable *and* preserves its attributes (device, dtype, methods, ...).

    After each call it enforces the FixAtoms constraints on the provided `state`.
    """
    def __init__(self, model, fix):
        self._model = model
        self._fix = fix

        # Prefer copying frequently-used attributes directly so attribute access like model.device works fast:
        for attr in ("device", "dtype", "compute_forces", "compute_stress"):
            if hasattr(model, attr):
                setattr(self, attr, getattr(model, attr))

    def __call__(self, state, *args, **kwargs):
        # Call the underlying model exactly as before
        res = self._model(state, *args, **kwargs)

        # Apply constraints to the state (zero forces/velocities, mask virials, recompute stress if possible)
        # apply_constraints_to_state is defined earlier in this module
        apply_constraints_to_state(state, self._fix)

        return res

    def __getattr__(self, name):
        # Proxy any other attribute or method to the underlying model
        # Note: __getattr__ is only invoked if attribute not found on this object,
        # so direct copies (device/dtype) above will not be proxied here.
        return getattr(self._model, name)

    def to(self, *args, **kwargs):
        # If someone calls wrapped_model.to(device) we'd like to forward to the underlying model,
        # and keep the proxy attributes up-to-date (device/dtype).
        result = None
        if hasattr(self._model, "to"):
            result = self._model.to(*args, **kwargs)
        # If the underlying model changed device/dtype, update our own attributes if present
        if hasattr(self._model, "device"):
            self.device = getattr(self._model, "device")
        if hasattr(self._model, "dtype"):
            self.dtype = getattr(self._model, "dtype")
        return result


def wrap_model(orig_model: Callable[[Any], Any], fix: FixAtoms):
    """Return a callable proxy object that preserves attributes required by torch-sim."""
    return ModelProxy(orig_model, fix)


def cell_volume_from_cell(cell: torch.Tensor) -> torch.Tensor:
    """
    Compute volume from a cell matrix. Accepts shapes:
      - (3,3) -> scalar
      - (B,3,3) -> (B,)
    """
    if cell is None:
        raise ValueError("cell is None")
    if cell.dim() == 2:
        return torch.abs(torch.det(cell))
    elif cell.dim() == 3:
        return torch.abs(torch.det(cell))
    else:
        raise ValueError(f"Unsupported cell shape: {cell.shape}")


def virials_to_stress(virials: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """
    Compute macroscopic stress from per-atom virials. Conventions vary by codebase;
    the most common expression is: stress = - (1/V) * sum(virials)

    virials accepted shapes: (N,3,3) or (B,N,3,3)
    cell accepted shapes: (3,3) or (B,3,3)

    returns shape: (3,3) or (B,3,3)
    """
    if virials is None:
        return None
    if virials.dim() == 3 and virials.shape[1:] == (3, 3):
        per_sample = False
        vir_sum = virials.sum(dim=0)
    elif virials.dim() == 4 and virials.shape[2:] == (3, 3):
        per_sample = True
        vir_sum = virials.sum(dim=1)  # (B,3,3)
    else:
        raise ValueError(f"Unsupported virials shape: {virials.shape}")

    vol = cell_volume_from_cell(cell)
    # Make vol broadcastable
    if isinstance(vol, torch.Tensor) and vol.dim() == 0:
        vol = vol.unsqueeze(0) if per_sample else vol

    stress = -vir_sum / vol.view(-1, 1, 1) if per_sample else -vir_sum / vol
    return stress


def mask_virials_and_recompute_stress(state, fix, virial_attr_names=("virials", "per_atom_virials")):
    """
    If the `state` carries per-atom virials, zero contributions from fixed atoms and
    attempt to recompute `state.stress` if the state has a cell (state.cell or state.box).

    This function is conservative: if per-atom virials are absent, it does nothing (but returns False).
    """
    for name in virial_attr_names:
        if hasattr(state, name):
            vir = getattr(state, name)
            if vir is None:
                continue
            vir_masked = fix.apply_to_virials(vir)
            setattr(state, name, vir_masked)

            # attempt to recompute state.stress if possible
            if hasattr(state, "cell") and state.cell is not None:
                new_stress = virials_to_stress(vir_masked, state.cell)
                state.stress = new_stress
            return True
    return False


def _get_per_sample_counts_from_state(state):
    """
    Try to find per-sample atom counts on the state.
    Returns a 1D LongTensor of length B or None if not found.
    Looks for common attribute names and for an atom mask.
    """
    for attr in ("n_atoms", "natoms", "atom_counts", "natoms_per_sample", "N_atoms"):
        if hasattr(state, attr):
            counts = getattr(state, attr)
            # Convert to 1D long tensor if possible
            try:
                counts_t = torch.as_tensor(counts, dtype=torch.long, device=state.positions.device)
                if counts_t.dim() == 0:
                    # single scalar -> not per-sample
                    continue
                return counts_t.view(-1)
            except Exception:
                continue
    # Try atom mask (common: state.atom_mask shape (B,N))
    if hasattr(state, "atom_mask"):
        mask = getattr(state, "atom_mask")
        if torch.is_tensor(mask) and mask.dim() == 2:
            return mask.sum(dim=1).to(dtype=torch.long, device=state.positions.device)
    return None


def apply_constraints_to_state(state, fix):
    """
    Enforce constraint effects on state:
      - zero forces on fixed atoms
      - zero velocities on fixed atoms
      - zero per-atom virials for fixed atoms (if present) and try to recompute state.stress

    This function supports three layouts:
      1) single-system: positions/forces shape (N,3)
      2) padded batched: (B,Nmax,3)
      3) flattened batched: (Sum_N,3) with per-sample counts in state (e.g., state.n_atoms / state.natoms)
    """
    dev = None
    if hasattr(state, "positions") and state.positions is not None:
        dev = state.positions.device

    # Helper to zero flattened global indices
    def _zero_global_indices(tensor, global_idx):
        if global_idx.numel() == 0:
            return tensor
        tensor = tensor.clone()
        if tensor.dim() == 2:
            tensor[global_idx] = 0.0
        elif tensor.dim() == 3:
            # this should not happen for flattened layout, but keep for safety
            tensor[:, global_idx, :] = 0.0
        return tensor

    # 1) HANDLE FORCES
    if hasattr(state, "forces") and state.forces is not None:
        forces = state.forces
        if forces.dim() == 3:
            # batched padded (B,N,3) -> use existing fix helper
            state.forces = fix.apply_to_forces(forces)
        elif forces.dim() == 2:
            # Could be single-system (N,3) OR flattened (Sum_N,3)
            counts = _get_per_sample_counts_from_state(state)
            if counts is None:
                # single system
                state.forces = fix.apply_to_forces(forces)
            else:
                # flattened layout: zero per-sample global indices
                offsets = torch.cumsum(torch.cat([torch.tensor([0], device=forces.device, dtype=torch.long), counts[:-1]]), dim=0)
                idx = fix.fix_indices.to(forces.device)
                out = forces.clone()
                B = counts.numel()
                for i in range(B):
                    ci = int(counts[i].item())
                    if ci <= 0:
                        continue
                    valid = idx[idx < ci]
                    if valid.numel() == 0:
                        continue
                    gidx = offsets[i] + valid
                    out[gidx] = 0.0
                state.forces = out
        else:
            # Unhandled forces layout
            raise ValueError(f"Unsupported forces tensor shape: {state.forces.shape}")

    # 2) HANDLE VELOCITIES (similar logic)
    if hasattr(state, "velocities") and state.velocities is not None:
        vel = state.velocities
        if vel.dim() == 3:
            state.velocities = fix.zero_velocities(vel)
        elif vel.dim() == 2:
            counts = _get_per_sample_counts_from_state(state)
            if counts is None:
                state.velocities = fix.zero_velocities(vel)
            else:
                offsets = torch.cumsum(torch.cat([torch.tensor([0], device=vel.device, dtype=torch.long), counts[:-1]]), dim=0)
                idx = fix.fix_indices.to(vel.device)
                out = vel.clone()
                B = counts.numel()
                for i in range(B):
                    ci = int(counts[i].item())
                    if ci <= 0:
                        continue
                    valid = idx[idx < ci]
                    if valid.numel() == 0:
                        continue
                    gidx = offsets[i] + valid
                    out[gidx] = 0.0
                state.velocities = out
        else:
            raise ValueError(f"Unsupported velocities tensor shape: {state.velocities.shape}")

    # 3) HANDLE PER-ATOM VIRIALS (and recompute stress)
    # We attempt to find per-atom virials on common attribute names, then mask them similarly.
    for name in ("virials", "per_atom_virials", "atomic_virials"):
        if hasattr(state, name):
            vir = getattr(state, name)
            if vir is None:
                continue
            # vir could be (N,3,3), (B,N,3,3), (Sum_N,3,3) or the 6-component versions
            if vir.dim() == 4 and vir.shape[2:] == (3, 3):
                # (B,N,3,3)
                state_vir_masked = fix.apply_to_virials(vir)
                setattr(state, name, state_vir_masked)
            elif vir.dim() == 3 and vir.shape[1:] == (3, 3):
                # Either (N,3,3) single-system OR flattened (Sum_N,3,3) — use counts detection
                counts = _get_per_sample_counts_from_state(state)
                if counts is None:
                    # single system
                    state_vir_masked = fix.apply_to_virials(vir)
                    setattr(state, name, state_vir_masked)
                else:
                    offsets = torch.cumsum(torch.cat([torch.tensor([0], device=vir.device, dtype=torch.long), counts[:-1]]), dim=0)
                    out = vir.clone()
                    idx = fix.fix_indices.to(vir.device)
                    B = counts.numel()
                    for i in range(B):
                        ci = int(counts[i].item())
                        if ci <= 0:
                            continue
                        valid = idx[idx < ci]
                        if valid.numel() == 0:
                            continue
                        gidx = offsets[i] + valid
                        out[gidx, :, :] = 0.0
                    setattr(state, name, out)
            elif vir.dim() == 2 and vir.shape[1] == 6:
                # (N,6) single system
                state_vir_masked = fix.apply_to_virials(vir)
                setattr(state, name, state_vir_masked)
            elif vir.dim() == 3 and vir.shape[2] == 6:
                # (B,N,6)
                state_vir_masked = fix.apply_to_virials(vir)
                setattr(state, name, state_vir_masked)
            # attempt to recompute macroscopic stress if state.cell exists
            if hasattr(state, "cell") and state.cell is not None:
                try:
                    from .utils import virials_to_stress
                    state.stress = virials_to_stress(getattr(state, name), state.cell)
                except Exception:
                    # best-effort: don't fail if recompute fails
                    pass
            break

    return state


def wrap_integrator_step(step_fn: Callable[..., Any], fix: FixAtoms) -> Callable[..., Any]:
    """
    Wrap integrator step to:
      - save fixed-atom positions (handles padded-batched and flattened layouts),
      - call original step_fn,
      - apply constraints,
      - restore fixed positions.
    """
    def wrapped(*args, **kwargs):
        state = None
        if len(args) >= 1:
            state = args[0]
        elif "state" in kwargs:
            state = kwargs["state"]

        if state is None:
            return step_fn(*args, **kwargs)

        # Prepare saved fixed positions + bookkeeping
        positions = state.positions
        dev = positions.device
        idx = fix.fix_indices.to(dev)

        prev_fixed_info = {"layout": None, "per_sample": None, "offsets": None}
        # Detect padded batched (B,N,3)
        if positions.dim() == 3:
            prev_valid = idx[idx < positions.shape[1]]
            if prev_valid.numel() > 0:
                prev_fixed = positions[:, prev_valid, :].clone()  # (B, num_valid, 3)
            else:
                prev_fixed = torch.empty((positions.shape[0], 0, positions.shape[2]), device=dev, dtype=positions.dtype)
            prev_fixed_info["layout"] = "padded"
            prev_fixed_info["per_sample"] = (prev_valid,)
            prev_fixed_info["prev_fixed"] = prev_fixed

        # Single system (N,3) or flattened (Sum_N,3) — need to check for per-sample counts
        elif positions.dim() == 2:
            counts = None
            for attr in ("n_atoms", "natoms", "atom_counts", "natoms_per_sample", "N_atoms"):
                if hasattr(state, attr):
                    counts = getattr(state, attr)
                    try:
                        counts = torch.as_tensor(counts, dtype=torch.long, device=dev).view(-1)
                    except Exception:
                        counts = None
                    break
            if counts is None and hasattr(state, "atom_mask"):
                mask = getattr(state, "atom_mask")
                if torch.is_tensor(mask) and mask.dim() == 2:
                    counts = mask.sum(dim=1).to(dtype=torch.long, device=dev)

            if counts is None:
                # single system (N,3)
                valid = idx[idx < positions.shape[0]]
                if valid.numel() > 0:
                    prev_fixed = positions[valid].clone()  # (num_valid,3)
                else:
                    prev_fixed = torch.empty((0, positions.shape[1]), device=dev, dtype=positions.dtype)
                prev_fixed_info["layout"] = "single"
                prev_fixed_info["per_sample"] = (valid,)
                prev_fixed_info["prev_fixed"] = prev_fixed
            else:
                # flattened layout: positions shape (Sum_N,3)
                counts = counts.to(device=dev, dtype=torch.long)
                offsets = torch.cumsum(torch.cat([torch.tensor([0], device=dev, dtype=torch.long), counts[:-1]]), dim=0)
                B = counts.numel()
                per_sample_valid = []
                per_sample_prev = []
                for i in range(B):
                    ci = int(counts[i].item())
                    if ci <= 0:
                        per_sample_valid.append(torch.tensor([], dtype=torch.long, device=dev))
                        per_sample_prev.append(torch.empty((0, positions.shape[1]), device=dev, dtype=positions.dtype))
                        continue
                    valid = idx[idx < ci]
                    if valid.numel() == 0:
                        per_sample_valid.append(valid)
                        per_sample_prev.append(torch.empty((0, positions.shape[1]), device=dev, dtype=positions.dtype))
                        continue
                    gidx = offsets[i] + valid
                    per_sample_valid.append(valid)
                    per_sample_prev.append(positions[gidx].clone())  # (num_valid,3)
                prev_fixed_info["layout"] = "flattened"
                prev_fixed_info["offsets"] = offsets
                prev_fixed_info["per_sample"] = per_sample_valid
                prev_fixed_info["prev_fixed"] = per_sample_prev
        else:
            raise ValueError(f"Unsupported positions shape: {positions.shape}")

        # Call the real integrator step
        res = step_fn(*args, **kwargs)

        # Apply constraints (zero forces/velocities/virials and recompute stress when possible)
        apply_constraints_to_state(state, fix)

        # Restore fixed positions according to layout
        if prev_fixed_info["layout"] == "padded":
            valid, = prev_fixed_info["per_sample"]
            state.positions[:, valid, :] = prev_fixed_info["prev_fixed"]
        elif prev_fixed_info["layout"] == "single":
            valid, = prev_fixed_info["per_sample"]
            if valid.numel() > 0:
                state.positions[valid] = prev_fixed_info["prev_fixed"]
        elif prev_fixed_info["layout"] == "flattened":
            offsets = prev_fixed_info["offsets"]
            per_valid = prev_fixed_info["per_sample"]
            per_prev = prev_fixed_info["prev_fixed"]
            for i in range(len(per_valid)):
                valid = per_valid[i]
                if valid.numel() == 0:
                    continue
                gidx = offsets[i] + valid.to(device=dev)
                state.positions[gidx] = per_prev[i]
        else:
            # nothing to restore
            pass

        return res

    return wrapped


def wrap_optimizer_step(opt_step_fn: Callable[..., Any], fix: FixAtoms) -> Callable[..., Any]:
    """
    Wrap optimizer step/run to enforce constraints after each call.
    Same robust handling as wrap_integrator_step for padded/flattened/single layouts.
    """
    def wrapped(*args, **kwargs):
        state = None
        if len(args) >= 1:
            state = args[0]
        elif "state" in kwargs:
            state = kwargs["state"]

        if state is None:
            return opt_step_fn(*args, **kwargs)

        # Save fixed positions (reuse the same logic as integrator wrapper)
        positions = state.positions
        dev = positions.device
        idx = fix.fix_indices.to(dev)

        prev_fixed_info = {"layout": None, "per_sample": None, "offsets": None}
        if positions.dim() == 3:
            prev_valid = idx[idx < positions.shape[1]]
            if prev_valid.numel() > 0:
                prev_fixed = positions[:, prev_valid, :].clone()
            else:
                prev_fixed = torch.empty((positions.shape[0], 0, positions.shape[2]), device=dev, dtype=positions.dtype)
            prev_fixed_info["layout"] = "padded"
            prev_fixed_info["per_sample"] = (prev_valid,)
            prev_fixed_info["prev_fixed"] = prev_fixed
        elif positions.dim() == 2:
            counts = None
            for attr in ("n_atoms", "natoms", "atom_counts", "natoms_per_sample", "N_atoms"):
                if hasattr(state, attr):
                    counts = getattr(state, attr)
                    try:
                        counts = torch.as_tensor(counts, dtype=torch.long, device=dev).view(-1)
                    except Exception:
                        counts = None
                    break
            if counts is None and hasattr(state, "atom_mask"):
                mask = getattr(state, "atom_mask")
                if torch.is_tensor(mask) and mask.dim() == 2:
                    counts = mask.sum(dim=1).to(dtype=torch.long, device=dev)

            if counts is None:
                valid = idx[idx < positions.shape[0]]
                if valid.numel() > 0:
                    prev_fixed = positions[valid].clone()
                else:
                    prev_fixed = torch.empty((0, positions.shape[1]), device=dev, dtype=positions.dtype)
                prev_fixed_info["layout"] = "single"
                prev_fixed_info["per_sample"] = (valid,)
                prev_fixed_info["prev_fixed"] = prev_fixed
            else:
                counts = counts.to(device=dev, dtype=torch.long)
                offsets = torch.cumsum(torch.cat([torch.tensor([0], device=dev, dtype=torch.long), counts[:-1]]), dim=0)
                B = counts.numel()
                per_sample_valid = []
                per_sample_prev = []
                for i in range(B):
                    ci = int(counts[i].item())
                    if ci <= 0:
                        per_sample_valid.append(torch.tensor([], dtype=torch.long, device=dev))
                        per_sample_prev.append(torch.empty((0, positions.shape[1]), device=dev, dtype=positions.dtype))
                        continue
                    valid = idx[idx < ci]
                    if valid.numel() == 0:
                        per_sample_valid.append(valid)
                        per_sample_prev.append(torch.empty((0, positions.shape[1]), device=dev, dtype=positions.dtype))
                        continue
                    gidx = offsets[i] + valid
                    per_sample_valid.append(valid)
                    per_sample_prev.append(positions[gidx].clone())
                prev_fixed_info["layout"] = "flattened"
                prev_fixed_info["offsets"] = offsets
                prev_fixed_info["per_sample"] = per_sample_valid
                prev_fixed_info["prev_fixed"] = per_sample_prev
        else:
            raise ValueError(f"Unsupported positions shape: {positions.shape}")

        # Call optimizer
        res = opt_step_fn(*args, **kwargs)

        # Apply constraints
        apply_constraints_to_state(state, fix)

        # Restore positions
        if prev_fixed_info["layout"] == "padded":
            valid, = prev_fixed_info["per_sample"]
            state.positions[:, valid, :] = prev_fixed_info["prev_fixed"]
        elif prev_fixed_info["layout"] == "single":
            valid, = prev_fixed_info["per_sample"]
            if valid.numel() > 0:
                state.positions[valid] = prev_fixed_info["prev_fixed"]
        elif prev_fixed_info["layout"] == "flattened":
            offsets = prev_fixed_info["offsets"]
            per_valid = prev_fixed_info["per_sample"]
            per_prev = prev_fixed_info["prev_fixed"]
            for i in range(len(per_valid)):
                valid = per_valid[i]
                if valid.numel() == 0:
                    continue
                gidx = offsets[i] + valid.to(device=dev)
                state.positions[gidx] = per_prev[i]
        return res

    return wrapped
