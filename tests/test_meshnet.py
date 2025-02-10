import os
import torch
import pytest
from app.code.executor.meshnet import enMesh_checkpoint

def test_modelAE_from_production_config():
    # Emulate production: get the modelAE.json from app/code/executor/
    # Import the package to use its __file__ attribute.
    import app.code.executor  
    config_file_path = os.path.join(os.path.dirname(app.code.executor.__file__), "modelAE.json")
    
    # Instantiate the model using production parameters.
    model = enMesh_checkpoint(
        in_channels=1,
        n_classes=3,
        channels=5,
        config_file=config_file_path
    )
    
    # Create a small random input tensor.
    x = torch.randn(1, 1, 16, 16, 16)
    
    # Test in train mode (which uses checkpoint_sequential)
    model.train()
    output_train = model(x)
    expected_shape = (1, 3, 16, 16, 16)  # Expecting n_classes=3 as output channels.
    assert output_train.shape == expected_shape, f"Train mode output shape mismatch: expected {expected_shape}, got {output_train.shape}"
    
    # Test in eval mode (using inference mode)
    model.eval()
    with torch.no_grad():
        output_eval = model(x)
    assert output_eval.shape == expected_shape, f"Eval mode output shape mismatch: expected {expected_shape}, got {output_eval.shape}"

