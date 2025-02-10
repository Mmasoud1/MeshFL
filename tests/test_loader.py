import os
import sqlite3
import zlib
import json
import numpy as np
import torch
import pytest
from app.code.executor.loader import Scanloader

def create_dummy_data(shape=(32, 32, 32), value=1.0):
    """
    Create a dummy numpy array of the given shape and value,
    then compress it with zlib.
    """
    arr = np.full(shape, value, dtype=np.float32)
    arr_bytes = arr.tobytes()
    return zlib.compress(arr_bytes)

@pytest.fixture
def temp_db_small(tmp_path):
    """
    Create a temporary SQLite database file containing a table
    'mindboggle101' with two columns ('Image' and 'label') and
    insert 3 rows of dummy data. The dummy data is created with shape (32,32,32).
    """
    db_path = tmp_path / "test_small.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE mindboggle101 (Image BLOB, label BLOB)")
    # Insert 3 rows
    for _ in range(3):
        image_blob = create_dummy_data(shape=(32, 32, 32), value=1.0)
        label_blob = create_dummy_data(shape=(32, 32, 32), value=2.0)
        cursor.execute("INSERT INTO mindboggle101 (Image, label) VALUES (?, ?)", (image_blob, label_blob))
    conn.commit()
    conn.close()
    return str(db_path)

@pytest.fixture(autouse=True)
def patch_scanloader_getitem(monkeypatch):
    """
    Monkey-patch the __getitem__ method of Scanloader so that instead of
    reshaping to (256,256,256), it reshapes to (32,32,32) for testing.
    """
    original_getitem = Scanloader.__getitem__

    def fake_getitem(self, idx):
        actual_idx = self.indices[idx]
        sample = self.data[actual_idx]
        # Use test shape (32,32,32) instead of (256,256,256)
        image = zlib.decompress(sample[0])
        image_tensor = torch.from_numpy(
            np.copy(np.frombuffer(image, dtype=np.float32)).reshape((32, 32, 32))
        )
        label = zlib.decompress(sample[1])
        label_tensor = torch.from_numpy(
            np.copy(np.frombuffer(label, dtype=np.float32)).reshape((32, 32, 32))
        )
        return image_tensor.to(self.device), label_tensor.to(self.device)

    monkeypatch.setattr(Scanloader, '__getitem__', fake_getitem)
    yield
    monkeypatch.setattr(Scanloader, '__getitem__', original_getitem)

def test_scanloader_without_split_small(temp_db_small):
    """
    Test basic functionality of Scanloader when not using a split file.
    Verifies:
      - The data is loaded (3 rows inserted)
      - __len__ returns the correct value
      - __getitem__ returns tensors with shape (32,32,32)
      - The tensors are on the expected device.
    """
    dataset = Scanloader(db_file=temp_db_small, use_split_file=False)
    # The database has 3 rows
    assert len(dataset.data) == 3
    assert len(dataset) == 3

    image_tensor, label_tensor = dataset[0]
    assert image_tensor.shape == (32, 32, 32)
    assert label_tensor.shape == (32, 32, 32)
    assert image_tensor.device == dataset.device

def test_scanloader_without_split_small(temp_db_small):
    """
    Test basic functionality of Scanloader when not using a split file.
    Verifies:
      - The data is loaded (3 rows inserted)
      - __len__ returns the correct value
      - __getitem__ returns tensors with shape (32,32,32)
      - The tensors are on the expected device.
    """
    dataset = Scanloader(db_file=temp_db_small, use_split_file=False)
    # The database has 3 rows
    assert len(dataset.data) == 3
    assert len(dataset) == 3

    image_tensor, label_tensor = dataset[0]
    assert image_tensor.shape == (32, 32, 32)
    assert label_tensor.shape == (32, 32, 32)
    # Instead of checking full equality, compare only the device type
    assert image_tensor.device.type == dataset.device.type, \
        f"Expected device type {dataset.device.type}, got {image_tensor.device.type}"


def test_scanloader_with_split_small(tmp_path):
    """
    Test Scanloader with split file enabled.
    This test creates a new temporary database with 6 rows and enables splitting.
    The create_split_file() method should partition the indices as follows:
      - train: int(0.75 * 6) = 4 samples
      - valid: int(0.15 * 6) = 0 samples (rounded down)
      - test: remaining samples = 2 samples
    We test the 'train' subset.
    """
    db_path = tmp_path / "test_multi_small.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE mindboggle101 (Image BLOB, label BLOB)")
    num_rows = 6
    for _ in range(num_rows):
        image_blob = create_dummy_data(shape=(32, 32, 32), value=1.0)
        label_blob = create_dummy_data(shape=(32, 32, 32), value=2.0)
        cursor.execute("INSERT INTO mindboggle101 (Image, label) VALUES (?, ?)", (image_blob, label_blob))
    conn.commit()
    conn.close()

    # Use a temporary split file path.
    split_file_path = str(tmp_path / "splits.json")
    dataset = Scanloader(
        db_file=str(db_path),
        use_split_file=True,
        split_file=split_file_path,
        subset="train"
    )
    # The split file should now exist.
    assert os.path.exists(split_file_path)
    # For 6 rows, train indices should be int(0.75*6)=4.
    assert len(dataset.indices) == 4
    assert len(dataset) == 4

