import pytest
import torch
from app.code.executor.dice import dice, faster_dice

def test_dice_identical_arrays():
    """
    If x and y are identical, dice score should be 1.0.
    """
    x = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
    y = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
    score = dice(x, y)
    assert score == 1.0, f"Expected 1.0, got {score}"

def test_dice_completely_different():
    """
    If x and y have no overlap, dice score should be 0.
    """
    x = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    score = dice(x, y)
    assert score == 0.0, f"Expected 0.0, got {score}"

def test_dice_single_label_faster_dice():
    """
    Test faster_dice with a single label in 'labels'.
    """
    x = torch.tensor([[1, 1],
                      [0, 0]], dtype=torch.float32)
    y = torch.tensor([[1, 1],
                      [1, 0]], dtype=torch.float32)
    labels = [1]
    # Intersection is 2. sum(x) = 2, sum(y) = 3 => dice = (2*2)/(2+3) = 4/5 = 0.8
    score = faster_dice(x, y, labels)
    assert torch.isclose(score, torch.tensor(0.8)), f"Expected 0.8, got {score}"

def test_dice_multi_label_faster_dice():
    """
    Test faster_dice with multiple labels.
    """
    x = torch.tensor([
        [0, 1, 2],
        [2, 2, 1]
    ], dtype=torch.int64)

    y = torch.tensor([
        [0, 1, 2],
        [2, 1, 1]
    ], dtype=torch.int64)

    labels = [0, 1, 2]
    # Expected dice for each label: [1.0, 0.8, 0.8]
    score = faster_dice(x, y, labels)
    expected = torch.tensor([1.0, 0.8, 0.8], dtype=torch.float32)
    assert torch.allclose(score, expected), f"Expected {expected}, got {score}"

def test_faster_dice_shape_mismatch():
    """
    Check that faster_dice raises AssertionError if x and y have different shapes.
    """
    x = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
    y = torch.tensor([[1, 1, 1], [0, 0, 1]], dtype=torch.float32)
    labels = [1]

    with pytest.raises(AssertionError) as excinfo:
        faster_dice(x, y, labels)
    assert "both inputs should have same size" in str(excinfo.value)

