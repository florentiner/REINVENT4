"""Chemically Advanced Template Search (CATS) similarity"""

from __future__ import annotations

__all__ = ["CATSDistance"]

from dataclasses import dataclass
from typing import List

import numpy as np

from reinvent.chemistry.conversions import Conversions
from reinvent.chemistry.similarity import Similarity
from ..component_results import ComponentResults
from ..add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    smiles: List[List[str]]


@add_tag("__component")
class CATSDistance:
    def __init__(self, params: Parameters):
        self.chem = Conversions()
        self.similarity = Similarity()
        self.cats_params = []

        for smilies in params.smiles:
            cats_mols = self.chem.smiles_to_CATS(smilies)

            if not cats_mols:
                raise RuntimeError(f"{__name__}: unable to convert any SMILES to CATS")

            self.cats_params.append(cats_mols)

    def __call__(self, smilies: List[str]) -> np.array:
        scores = []

        for cats_mols in self.cats_params:
            query_cats = self.chem.smiles_to_CATS(smilies)

            scores.extend(
                [
                    self.similarity.calculate_cats_batch(cats_mol, query_cats)
                    for cats_mol in cats_mols
                ]
            )

        return ComponentResults(scores)