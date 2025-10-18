#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, Union
import torch
import torch_sim as ts
from torch_sim.typing import StateDict
from torch_sim.models.mace import MaceModel


class MaceModelWrapper(MaceModel):
    """
    Wrapper around torch_sim.models.mace.MaceModel that implements ASE-style
    FixAtoms constraints.

    Usage:
        wrapper = MaceModelWrapper(..., fix_indices=...)
        results = wrapper.forward(state)                       # uses constructor fix_indices
        results = wrapper.forward(state, fix_indices=new_idx)  # override for this call

    fix_indices may be:
      - 1D LongTensor of global atom indices (same ordering as sim_state.positions), or
      - 1D BoolTensor mask of length n_atoms, or
      - a list/ndarray of indices (converted internally).
    """

    def __init__(
        self,
        model: Union[str, torch.nn.Module, None] = None,
        *,
        fix_indices: Optional[torch.Tensor] = None,
        # all other kwargs forwarded to the parent MaceModel
        **mace_kwargs,
    ) -> None:
        # initialize base MaceModel with remaining kwargs
        super().__init__(model, **mace_kwargs)

        # Normalise and store constructor fix indices (kept on device used by model)
        if fix_indices is None:
            self._fix_indices: Optional[torch.Tensor] = None
        else:
            if not isinstance(fix_indices, torch.Tensor):
                t = torch.tensor(fix_indices)
            else:
                t = fix_indices
            # keep boolean masks as bool, otherwise convert to long
            if t.dtype == torch.bool:
                self._fix_indices = t.to(device=self.device)
            else:
                self._fix_indices = t.to(device=self.device).long()

        # Storage for original fixed positions (captured on first forward)
        self._fixed_positions: Optional[torch.Tensor] = None

    def _normalize_fix_idx(
        self, fix_input: Optional[torch.Tensor], n_atoms: int
    ) -> Optional[torch.Tensor]:
        """Return 1D LongTensor of valid global indices on self.device, or None."""
        if fix_input is None:
            return None
        t = fix_input
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.to(device=self.device)
        if t.dtype == torch.bool:
            t = torch.nonzero(t, as_tuple=False).view(-1)
        else:
            t = t.view(-1).long()
        # clip / validate indices
        if t.numel() == 0:
            return None
        valid_mask = (t >= 0) & (t < n_atoms)
        if not valid_mask.all():
            t = t[valid_mask]
            if t.numel() == 0:
                return None
        return t

    def forward(
        self,
        state: ts.SimState | StateDict,
        fix_indices: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with ASE FixAtoms semantics.

        Parameters
        ----------
        state : SimState | dict
            The simulation state (same format expected by original MaceModel).
        fix_indices : Optional[torch.Tensor]
            If provided, overrides constructor fix indices for this call.
        """
        # Build canonical sim_state as in the original MaceModel (so we can inspect positions)
        sim_state = (
            state
            if isinstance(state, ts.SimState)
            else ts.SimState(**state, masses=torch.ones_like(state["positions"]))
        )

        # Determine effective fix indices: forward arg overrides constructor value
        provided_fix = fix_indices if fix_indices is not None else getattr(self, "_fix_indices", None)
        # normalize to LongTensor of global indices on the model device (or None)
        fix_idx = self._normalize_fix_idx(provided_fix, n_atoms=sim_state.positions.shape[0])

        # If we have fixed atoms and haven't captured their original positions yet,
        # capture them now (ASE behavior: capture positions when constraint is set)
        if fix_idx is not None and self._fixed_positions is None:
            # store original positions (on model device)
            self._fixed_positions = sim_state.positions[fix_idx].detach().clone().to(device=self.device)

        # Prepare adjusted positions that enforce fixed-atom original positions
        positions_adj = sim_state.positions.clone()
        if fix_idx is not None and self._fixed_positions is not None and fix_idx.numel() > 0:
            positions_adj[fix_idx, :] = self._fixed_positions

        # Build a ts.SimState object with adjusted positions and call parent forward.
        # This prevents the parent from converting a dict to SimState
        original_positions = sim_state.positions
        sim_state.positions = positions_adj
        results = super().forward(sim_state)

        # Ensure forces/stress tensors are writable copies before we modify them
        if "forces" in results and results["forces"] is not None:
            results["forces"] = results["forces"].detach().clone()
        if "stress" in results and results["stress"] is not None:
            results["stress"] = results["stress"].detach().clone()

        # Apply ASE-style adjust_forces: zero forces on fixed atoms
        if fix_idx is not None and fix_idx.numel() > 0 and "forces" in results and results["forces"] is not None:
            forces = results["forces"]
            # valid_mask should already be ensured, but check again to avoid indexing errors
            valid_mask = (fix_idx >= 0) & (fix_idx < forces.shape[0])
            if valid_mask.any():
                valid_idx = fix_idx[valid_mask]
                forces[valid_idx, :] = forces.new_zeros((valid_idx.numel(), forces.shape[1]))
                results["forces"] = forces

        # Zero stress for any system that contains a fixed atom (simple and conservative)
        if (
            fix_idx is not None
            and fix_idx.numel() > 0
            and "stress" in results
            and results["stress"] is not None
        ):
            # map fixed atoms to their systems (sim_state.system_idx is global mapping)
            sys_idx_of_fixed = sim_state.system_idx[fix_idx].to(device=results["stress"].device)
            unique_sys = torch.unique(sys_idx_of_fixed)
            if unique_sys.numel() > 0:
                # zero the 3x3 stress for each affected system
                results["stress"][unique_sys, :, :] = results["stress"].new_zeros(
                    (
                        unique_sys.numel(),
                        results["stress"].shape[1],
                        results["stress"].shape[2],
                    )
                )

        return results

