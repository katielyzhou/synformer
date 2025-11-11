import abc
import dataclasses
from typing import TypeAlias

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import BRICS

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.mol import Molecule, FingerprintOption

LossDict: TypeAlias = dict[str, torch.Tensor]
AuxDict: TypeAlias = dict[str, torch.Tensor]


@dataclasses.dataclass
class ReactantRetrievalResult:
    reactants: np.ndarray
    fingerprint_predicted: np.ndarray
    fingerprint_retrieved: np.ndarray
    distance: np.ndarray
    indices: np.ndarray


class FragmentHead:

    def __init__(self, fingerprint_dim: int):
        super().__init__()
        self._fingerprint_dim = fingerprint_dim
        self._fp_option = FingerprintOption(type="morgan", morgan_radius=2, morgan_n_bits=2048)

    @property
    def fingerprint_dim(self) -> int:
        return self._fingerprint_dim

    def predict(self, h: torch.Tensor, target_mol: Molecule) -> torch.Tensor:
        """
        Return a list of candidate fragment fingerprints to consider for building blocks.
        """
        fragments=BRICS.BRICSDecompose(target_mol._rdmol)

        frag_mol = [Molecule.from_rdmol(Chem.MolFromSmiles(frag)) for frag in fragments]
        frag_fp = [i.get_fingerprint(self._fp_option) for i in frag_mol] # returns np arrays

        fp_tensor = torch.stack([torch.tensor(fp, dtype=torch.float32) for fp in frag_fp], dim=0) # (n_fps, fp_dim)

        # Add batch dimension
        batch_shape = h.shape[:-1]
        fp_tensor = fp_tensor.unsqueeze(0).expand(*batch_shape, fp_tensor.shape[0], fp_tensor.shape[1])
        # (*batch, n_fps, fp_dim)

        return fp_tensor

    def retrieve_reactants(
        self,
        h: torch.Tensor,
        fpindex: FingerprintIndex,
        target_mol: Molecule,
        topk: int = 4,
    ) -> ReactantRetrievalResult:
        """
        Retrieve reactants based on fragments.
    
        Args:
            h:  Tensor of shape (*batch, h_dim).
            fpindex:  FingerprintIndex
            topk:  Number of reactants to retrieve per fingerprint.
        Returns:
            - numpy Molecule array of shape (*batch, n_fps, topk).
            - numpy fingerprint array of shape (*batch, n_fps, topk, fp_dim).
        """

        fp = self.predict(h, target_mol)  # (*batch, n_fps, fp_dim) ; fragments fp
        fp_dim = fp.shape[-1]
        out = np.empty(list(fp.shape[:-1]) + [topk], dtype=Molecule)
        out_fp = np.empty(list(fp.shape[:-1]) + [topk, fp_dim], dtype=np.float32)
        out_fp_pred = fp[..., None, :].expand(*out_fp.shape)
        out_dist = np.empty(list(fp.shape[:-1]) + [topk], dtype=np.float32)
        out_idx = np.empty(list(fp.shape[:-1]) + [topk], dtype=np.int64)

        fp_flat = fp.reshape(-1, fp_dim)
        out_flat = out.reshape(-1, topk)
        out_fp_flat = out_fp.reshape(-1, topk, fp_dim)
        out_dist_flat = out_dist.reshape(-1, topk)
        out_idx_flat = out_idx.reshape(-1, topk)
        
        # Searches FP catalogue for BB with nearest FP sim to fragment FP
        query_res = fpindex.query_cuda(q=fp_flat, k=topk)
        for i, q_res_subl in enumerate(query_res):
            for j, q_res in enumerate(q_res_subl):
                out_flat[i, j] = q_res.molecule
                out_fp_flat[i, j] = q_res.fingerprint
                out_dist_flat[i, j] = q_res.distance
                out_idx_flat[i, j] = q_res.index

        return ReactantRetrievalResult(
            reactants=out,
            fingerprint_predicted=out_fp_pred.detach().cpu().numpy(),
            fingerprint_retrieved=out_fp,
            distance=out_dist,
            indices=out_idx,
        )
