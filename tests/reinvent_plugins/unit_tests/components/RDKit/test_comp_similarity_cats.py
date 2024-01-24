import pytest

import numpy as np

from reinvent_plugins.components.RDKit.comp_similarity_cats import (
    Parameters,
    CATSDistance,
)


@pytest.mark.parametrize(
    "smiles, expected_results",
    [
        (
            ["c1ccccc1", "Cc1ccccc1"],
            [0, 0, 0, 0],
        ),
        (
            ["c1ccccc1", "CCCCCCCCCC"],
            [0, 31.36877428, 31.36877428, 0],
        ),
    ],
)
def test_comp_similarity_cats(smiles, expected_results):
    params = Parameters(
        [smiles]
    )
    td = CATSDistance(params)
    results = np.concatenate(td(smiles).scores)
    print(results)
    assert np.allclose(results, expected_results)
