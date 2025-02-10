import os
import tempfile
import torch
import logging
import numpy as np
import pytest
from app.code.executor.dist import training, GenericLogger

# ----------------------------
# Dummy implementations
# ----------------------------

# Dummy MeshNet: A minimal network that uses a single Conv3d.
class DummyMeshNet(torch.nn.Module):
    def __init__(self, in_channels, n_classes, channels, config_file):
        super(DummyMeshNet, self).__init__()
        # A single convolution that preserves spatial dimensions.
        self.conv = torch.nn.Conv3d(in_channels, n_classes, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
# We use DummyMeshNet as our enMesh_checkpoint.
DummyMeshNetCheckpoint = DummyMeshNet

# Dummy Loader: Returns a DataLoader from a simple dataset.
class DummyLoader:
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, length=4):
            self.length = length
            self.device = torch.device("cpu")
        def __len__(self):
            return self.length
        def __getitem__(self, idx):
            # Use a small shape for testing purposes.
            # The training class expects a tensor reshaped to (-1, 1, shape, shape, shape),
            # where shape = 256 // cubes. For cubes=1, shape is 256, but we use a smaller value.
            shape = 8
            # Create a dummy image (with an extra channel dimension) and label.
            image = torch.ones((1, shape, shape, shape))
            label = torch.zeros((shape, shape, shape), dtype=torch.long)
            return image, label

    def Scanloader(self, db_file, label_type, num_cubes):
        # Return a dummy Scanloader instance with a get_loaders method.
        class DummyScanloader:
            def __init__(self):
                self.device = torch.device("cpu")
            def get_loaders(self, batch_size=1, shuffle=True, num_workers=0):
                dataset = DummyLoader.DummyDataset()
                loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
                # For simplicity, return the same loader for train, valid, and test.
                return loader, loader, loader
        return DummyScanloader()

# Dummy Dice: Returns a constant tensor.
class DummyDice:
    @staticmethod
    def faster_dice(x, y, labels, fudge_factor=1e-8):
        # Return a dummy tensor (for example, a single value).
        return torch.tensor([1.0])

# Dummy Logger: Does nothing (or prints) for testing.
class DummyLogger:
    def log_message(self, message, level=logging.INFO):
        # For testing, we can simply pass or print(message)
        pass

# ----------------------------
# Fixture for training class setup
# ----------------------------
@pytest.fixture
def dummy_training_setup(tmp_path):
    """
    Create a temporary environment with a dummy modelAE.json file,
    and instantiate the training class with dummy dependencies.
    """
    # Create a temporary directory for the training "path"
    temp_dir = tmp_path / "dummy_dir"
    temp_dir.mkdir()

    # Create a dummy modelAE.json file (content is irrelevant for the dummy model)
    modelAE_path = temp_dir / "modelAE.json"
    with open(modelAE_path, "w") as f:
        f.write("{}")  # empty JSON since DummyMeshNet doesn't use it

    # Create a dummy output directory for saving the model
    output_dir = temp_dir / "output"
    output_dir.mkdir()

    # Use a dummy database file name; DummyLoader won't actually access it.
    dummy_db = "dummy.db"

    # Instantiate the training class.
    # Change cubes to 32 so that self.shape = 256 // 32 = 8,
    # which matches the dummy data's shape.
    train_instance = training(
        path=str(temp_dir),
        output_path=str(output_dir),
        logger=DummyLogger(),
        databasefile=dummy_db,
        loader=DummyLoader(),  # dummy loader instance
        meshnet=type("dummy_meshnet_module", (), {"enMesh_checkpoint": DummyMeshNetCheckpoint}),
        learning_rate=0.001,
        modelAE="modelAE.json",
        classes=3,
        Dice=DummyDice(),
        cubes=32  # Update this value!
    )
    return train_instance, temp_dir, output_dir


# ----------------------------
# Test functions for training class
# ----------------------------

def test_training_get_train(dummy_training_setup):
    """
    Test that get_train() returns a tuple (image, label) with expected tensor shapes.
    """
    train_instance, _, _ = dummy_training_setup
    image, label = train_instance.get_train()
    # The dummy dataset returns image of shape (1, 8, 8, 8) and label of shape (8, 8, 8).
    # After batching, shapes become (batch, ...) – we used batch_size=1.
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert image.shape[1:] == (1, 8, 8, 8)
    assert label.shape[1:] == (8, 8, 8)

def test_training_get_gradients(dummy_training_setup):
    """
    Test that get_gradients() returns gradients as a list of nested lists.
    """
    train_instance, _, _ = dummy_training_setup
    grads = train_instance.get_gradients()
    assert isinstance(grads, list)
    # Each parameter gradient should be a list.
    for grad in grads:
        assert isinstance(grad, list)

def test_training_save(dummy_training_setup):
    """
    Test that calling save() creates a file 'meshnet.pth' in the output directory.
    """
    train_instance, _, output_dir = dummy_training_setup
    train_instance.save()
    saved_file = os.path.join(str(output_dir), "meshnet.pth")
    assert os.path.exists(saved_file)

def test_training_agg_gradients(dummy_training_setup):
    """
    Test that agg_gradients() runs without error.
    """
    train_instance, _, _ = dummy_training_setup
    grads = train_instance.get_gradients()
    try:
        train_instance.agg_gradients(grads)
    except Exception as e:
        pytest.fail(f"agg_gradients raised an exception: {e}")

def test_training_validate(dummy_training_setup):
    """
    Test that validate() runs without error.
    """
    train_instance, _, _ = dummy_training_setup
    try:
        train_instance.validate()
    except Exception as e:
        pytest.fail(f"validate raised an exception: {e}")

