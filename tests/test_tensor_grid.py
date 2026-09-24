from types import SimpleNamespace

import pytest
import torch

from p2i.analyze.inspection import RuntimeInspection


def inspect(value, **options):
    capture = RuntimeInspection(**options)
    capture.tensor(SimpleNamespace(id='t'), value)
    return capture.tensors['t']


def test_all_elements_contribute_to_rectangular_grid_without_raw_values():
    source = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    result = inspect(source, grid_size=2)
    assert result['values'] is None
    assert result['grid']['shape'] == [2, 2]
    assert result['grid']['values'] == [[2.5, 4.5], [10.5, 12.5]]
    assert result['grid']['finite_elements'] == 16
    assert result['grid']['source_shape'] == [4, 4]
    assert result['mean'] == 7.5


def test_leading_axes_are_explicitly_averaged_and_odd_sized_tiles_cover_every_element():
    source = torch.zeros(2, 4, 4)
    source[1].fill_(8)
    result = inspect(source, grid_size=2)
    assert result['grid']['reduced_axes'] == [0]
    assert result['grid']['spatial_axes'] == [1, 2]
    assert result['grid']['values'] == [[4.0, 4.0], [4.0, 4.0]]
    assert result['grid']['finite_elements'] == 32


def test_rectangular_source_preserves_axis_ratio_in_bounded_grid():
    square = inspect(torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32))['grid']
    tall = inspect(torch.arange(32, dtype=torch.float32)[:, None].expand(32, 16))['grid']
    wide = inspect(torch.ones(16, 32))['grid']
    assert square['shape'] == [32, 32]
    assert square['values'][0][0] == 0.0
    assert square['values'][-1][-1] == 1023.0
    assert tall['shape'] == [32, 16]
    assert wide['shape'] == [16, 32]
    assert tall['values'][0] == [0.0] * 16
    assert tall['values'][-1] == [31.0] * 16
    assert tall['finite_elements'] == 32 * 16


def test_scalar_vector_non_finite_and_safety_threshold():
    assert inspect(torch.tensor(2.0))['grid']['values'] == [[2.0]]
    vector = inspect(torch.arange(12, dtype=torch.float32), grid_size=4)
    assert vector['grid']['shape'] == [1, 4]
    assert vector['grid']['values'] == [[1.0, 4.0, 7.0, 10.0]]
    mixed = inspect(torch.tensor([[1.0, float('nan')], [float('inf'), 5.0]]), grid_size=2)
    assert mixed['grid']['values'] == [[1.0, None], [None, 5.0]]
    assert mixed['grid']['finite_elements'] == 2
    excluded = inspect(torch.ones(17), max_summary_elements=16)
    assert excluded['grid'] is None and excluded['values'] is None
    assert 'threshold' in excluded['message']
    with pytest.raises(ValueError):
        RuntimeInspection(grid_size=33)
