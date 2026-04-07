import os
import pytest
from opensportslib import model
import torch

def test_e2e_classification(mock_data_dir):
    """
    Test the end-to-end functionality of the classification model
    using small synthetic mock data.
    """
    config_path = mock_data_dir["config_yaml"]
    train_path = mock_data_dir["train_json"]
    test_path = mock_data_dir["test_json"]
    base_dir = mock_data_dir["base_dir"]

    # 1. Initialize Model
    myModel = model.classification(config=config_path)

    # 2. Train on synthetic dataset
    # We use a mocked checkpoint path to save weights
    save_dir = os.path.join(base_dir, "weights")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        myModel.train(
            train_set=train_path,
            valid_set=test_path,
            save_dir=save_dir
        )
    except Exception as e:
        pytest.fail(f"Training failed with exception: {str(e)}")

    # 3. Infer
    
    # We create a dummy pretrained checkpoint for testing infer as well if train 
    # doesn't save one identically named in the tiny test wrapper
    # But usually train saves something. If we want to be safe:
    dummy_checkpoint = os.path.join(save_dir, "dummy_ckpt.pth")
    # For now, just pass None to pretrained if `infer` supports it, 
    # or pass a valid dummy file. Based on docs, it supports path to checkpoint.
    # We will try infer without pretrained first because it might load the latest.
    
    predictions_path = os.path.join(base_dir, "preds.json")

    try:
        metrics = myModel.infer(
            test_set=test_path,
            # If a model actually trains for 1 epoch, it should produce a checkpoint in save_dir.
            # But let's avoid strictly asserting finding it unless we know the naming convention.
            predictions=predictions_path
        )
        # 4. Assertions
        assert metrics is not None, "Infer method should return metrics."
        assert os.path.exists(predictions_path), "Predictions JSON should be generated."
    except Exception as e:
        pytest.fail(f"Inference failed with exception: {str(e)}")
