import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "remap_rec_head", Path(__file__).parents[1] / "scripts" / "remap_rec_head.py"
)
remap_rec_head = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(remap_rec_head)


def _weights(n_chars: int) -> dict:
    """Fake checkpoint whose every output column/row holds its own index."""
    n_ctc, n_nrtr = n_chars + 2, n_chars + 6
    return {
        remap_rec_head.CTC_W: np.tile(np.arange(n_ctc, dtype=np.float32), (3, 1)),
        remap_rec_head.CTC_B: np.arange(n_ctc, dtype=np.float32),
        remap_rec_head.NRTR_EMB: np.tile(np.arange(n_nrtr, dtype=np.float32)[:, None], (1, 2)),
        remap_rec_head.NRTR_PRJ: np.tile(np.arange(n_nrtr, dtype=np.float32), (2, 1)),
        "backbone.w": np.ones(4, dtype=np.float32),
    }


def test_remap_copies_shared_chars_and_uses_base_letters():
    stock = ["1", "a", "â", "x"]
    new = ["1", "ẩ", "a", "ạ", "€"]
    out = remap_rec_head.remap(_weights(len(stock)), stock, new)

    # CTC: [blank] + dict + [space]; stock index of "a" is 2, of "â" is 3, space is 5.
    assert out[remap_rec_head.CTC_B].tolist() == [0, 1, 3, 2, 2, 2.5, 5]
    assert out[remap_rec_head.CTC_W].shape == (3, 7)
    # NRTR: 4 special tokens + dict + [space] + 1 extra output.
    assert out[remap_rec_head.NRTR_PRJ][0].tolist() == [0, 1, 2, 3, 4, 6, 5, 5, 5.5, 8, 9]
    assert out[remap_rec_head.NRTR_EMB][:, 0].tolist() == [0, 1, 2, 3, 4, 6, 5, 5, 5.5, 8, 9]
    assert out["backbone.w"].tolist() == [1, 1, 1, 1]


def test_source_char_strips_tone_only():
    stock = {"â": 0, "o": 1}
    assert remap_rec_head.source_char("ẩ", stock) == "â"
    assert remap_rec_head.source_char("ộ", stock) == "o"  # no "ô" in stock -> bare letter
    assert remap_rec_head.source_char("€", stock) is None
