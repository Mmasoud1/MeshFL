import os
import json
import tempfile
import torch
import pytest
import sqlite3
import zlib
import numpy as np

# Create dummy implementations for FLContext, Shareable, and Signal.
class DummyFLContext:
    def get_prop(self, key):
        # For our purposes, always return a dummy site name.
        return "dummy_site"
    def get_job_id(self):
        return "dummy_job"

class DummyShareable(dict):
    pass

class DummySignal:
    pass

# A fixture to create a temporary dummy modelAE.json and splits.json
@pytest.fixture
def dummy_files(tmp_path):
    # Create a dummy modelAE.json file with a minimal valid configuration.
    modelae_config = {
        "bnorm": True,
        "gelu": False,
        "dropout_p": 0,
        "layers": [
            {
                "in_channels": -1,
                "out_channels": 5,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1
            },
            {
                "in_channels": 5,
                "out_channels": -1,
                "kernel_size": 1,
                "stride": 1,
                "padding": 0
            }
        ]
    }
    modelae_file = tmp_path / "modelAE.json"
    modelae_file.write_text(json.dumps(modelae_config))
    
    # Create a dummy splits.json file; content is minimal since it might not be used.
    splits_file = tmp_path / "splits.json"
    splits_file.write_text(json.dumps({"train": [0], "valid": [0], "test": [0]}))
    
    return str(modelae_file), str(splits_file), tmp_path

# A fixture to monkey-patch file path resolution for MeshNetExecutor.
@pytest.fixture
def patch_executor_paths(monkeypatch, dummy_files):
    # dummy_files fixture returns (modelae_path, splits_path, dummy_dir)
    modelae_path, splits_path, dummy_dir = dummy_files
    dummy_dir_str = str(dummy_dir)  # ensure it's a string

    # Create a dummy SQLite database with the table "mindboggle101"
    dummy_db_path = os.path.join(dummy_dir_str, "mindboggle.db")
    conn = sqlite3.connect(dummy_db_path)
    cursor = conn.cursor()
    # Create the expected table with columns "Image" and "GWlabels"
    cursor.execute("CREATE TABLE mindboggle101 (Image BLOB, GWlabels BLOB)")
    # Insert a single dummy row.
    # create a dummy array of shape (256, 256, 256) for both image and label.
    dummy_image = np.zeros((256, 256, 256), dtype=np.float32)
    dummy_label = np.zeros((256, 256, 256), dtype=np.float32)
    blob_image = zlib.compress(dummy_image.tobytes())
    blob_label = zlib.compress(dummy_label.tobytes())
    cursor.execute("INSERT INTO mindboggle101 (Image, GWlabels) VALUES (?, ?)", (blob_image, blob_label))
    conn.commit()
    conn.close()

    # Monkey-patch os.path.join so that when MeshNetExecutor builds file paths,
    # if the last argument is "modelAE.json" or "splits.json", we return our dummy files.
    original_join = os.path.join
    def fake_join(*args):
        if args[-1] == "modelAE.json":
            return modelae_path
        if args[-1] == "splits.json":
            return splits_path
        return original_join(*args)
    monkeypatch.setattr(os.path, "join", fake_join)

    # Patch os.path.dirname to return our dummy directory.
    monkeypatch.setattr(os.path, "dirname", lambda path: dummy_dir_str)

    # Patch get_data_directory_path from the paths module to return our dummy directory.
    from app.code.executor import paths
    monkeypatch.setattr(paths, "get_data_directory_path", lambda fl_ctx: dummy_dir_str)

# Test that instantiates MeshNetExecutor and calls its execute method.
def test_executor_initialization(monkeypatch, patch_executor_paths):
    # Import the MeshNetExecutor from meshnet_executor.py
    from app.code.executor.meshnet_executor import MeshNetExecutor

    # Instantiate a dummy FLContext, Shareable, and Signal.
    dummy_fl_ctx = DummyFLContext()
    dummy_shareable = DummyShareable()
    dummy_shareable["aggregated_gradients"] = []  # For the accept_aggregated_gradients branch.
    dummy_signal = DummySignal()

    # Instantiate MeshNetExecutor.
    executor = MeshNetExecutor()

    # At this point, __init__ should have run without error (using our dummy modelAE.json and splits.json).
    # Call execute with a task that does not trigger data loader initialization.
    # Choose "accept_aggregated_gradients" which calls apply_gradients.
    result = executor.execute("accept_aggregated_gradients", dummy_shareable, dummy_fl_ctx, dummy_signal)

    # Check that the returned shareable is a dictionary .
    assert isinstance(result, dict)

