"""
Pytest configuration and fixtures
"""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device():
    """Get device for testing"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def dummy_image():
    """Create a dummy image for testing"""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def dummy_tensor():
    """Create a dummy tensor for testing"""
    return torch.randn(1, 3, 224, 224)

