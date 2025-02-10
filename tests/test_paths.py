import os
import pytest
from app.code.executor.paths import (
    get_data_directory_path,
    get_output_directory_path,
    get_parameters_file_path,
)

# ----------------------------
# Dummy FLContext for Testing
# ----------------------------
class DummyFLContext:
    def __init__(self, client_name, job_id="job123"):
        self._client_name = client_name
        self._job_id = job_id

    def get_prop(self, key):
        # We assume key is FLContextKey.CLIENT_NAME; return our dummy client name.
        return self._client_name

    def get_job_id(self):
        return self._job_id

@pytest.fixture
def dummy_context():
    # Use a test client name and a job id.
    return DummyFLContext(client_name="test_site", job_id="job123")

# ----------------------------
# Test for get_data_directory_path()
# ----------------------------
def test_get_data_directory_path(monkeypatch, tmp_path, dummy_context):
    """
    Test that get_data_directory_path returns the expected simulator path.
    The simulator path is constructed as:
       os.path.abspath(os.path.join(os.getcwd(), "../../../test_data", site_name))
    """
    # Set up a fake current working directory.
    fake_cwd = tmp_path / "subdir"
    fake_cwd.mkdir()
    monkeypatch.setattr(os, "getcwd", lambda: str(fake_cwd))

    # Compute expected simulator path.
    expected_simulator_path = os.path.abspath(
        os.path.join(str(fake_cwd), "../../../test_data", dummy_context.get_prop("CLIENT_NAME"))
    )
    # Create the expected directory so that os.path.exists returns True.
    os.makedirs(expected_simulator_path, exist_ok=True)

    result = get_data_directory_path(dummy_context)
    assert result == expected_simulator_path, f"Expected {expected_simulator_path}, got {result}"

# ----------------------------
# Test for get_output_directory_path()
# ----------------------------
def test_get_output_directory_path(monkeypatch, tmp_path, dummy_context):
    """
    Test that get_output_directory_path returns the correct simulator path and creates it if needed.
    The expected path is:
      os.path.abspath(os.path.join(os.getcwd(), "../../../test_output", job_id, site_name))
    """
    fake_cwd = tmp_path / "subdir"
    fake_cwd.mkdir()
    monkeypatch.setattr(os, "getcwd", lambda: str(fake_cwd))

    expected_output_path = os.path.abspath(
        os.path.join(str(fake_cwd), "../../../test_output", dummy_context.get_job_id(), dummy_context.get_prop("CLIENT_NAME"))
    )
    # Ensure the directory does not exist initially.
    if os.path.exists(expected_output_path):
        os.rmdir(expected_output_path)

    result = get_output_directory_path(dummy_context)
    # The function should create the directory.
    assert os.path.exists(result), "Output directory was not created."
    assert result == expected_output_path, f"Expected {expected_output_path}, got {result}"

# ----------------------------
# Tests for get_parameters_file_path()
# ----------------------------
def test_get_parameters_file_path_with_env(monkeypatch, tmp_path, dummy_context):
    """
    Test that if the environment variable PARAMETERS_FILE_PATH is set to an existing file,
    get_parameters_file_path returns that file.
    """
    dummy_param_file = tmp_path / "dummy_parameters.json"
    dummy_param_file.write_text('{"param": "value"}')
    monkeypatch.setenv("PARAMETERS_FILE_PATH", str(dummy_param_file))

    result = get_parameters_file_path(dummy_context)
    assert result == str(dummy_param_file), f"Expected {dummy_param_file}, got {result}"

def test_get_parameters_file_path_fallback(monkeypatch, tmp_path, dummy_context):
    """
    Test the fallback behavior for get_parameters_file_path() when the environment variable is not set.
    The simulator path is computed as:
       os.path.abspath(os.path.join(os.getcwd(), "../test_data", "server", "parameters.json"))
    """
    # Remove the environment variable if set.
    monkeypatch.delenv("PARAMETERS_FILE_PATH", raising=False)

    fake_cwd = tmp_path / "subdir"
    fake_cwd.mkdir()
    monkeypatch.setattr(os, "getcwd", lambda: str(fake_cwd))

    expected_simulator_file = os.path.abspath(
        os.path.join(str(fake_cwd), "../test_data", "server", "parameters.json")
    )
    # Ensure the directory exists.
    os.makedirs(os.path.dirname(expected_simulator_file), exist_ok=True)
    with open(expected_simulator_file, "w") as f:
        f.write('{"param": "value"}')

    result = get_parameters_file_path(dummy_context)
    assert result == expected_simulator_file, f"Expected {expected_simulator_file}, got {result}"

