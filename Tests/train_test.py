import pytest
import torch
from Train import ModelTrainer, EpochData, load_losses
import Train
from unittest.mock import mock_open, patch
from torch.utils.data import DataLoader, Dataset, random_split


class MockTrainDataloader(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return torch.randn(1, 96, 96), torch.randn(30)


class MockTestDataloader(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return torch.randn(1, 96, 96)


@pytest.fixture
def submission_format():
    return [
        {"RowId": 1, "ImageId": 1, "Feature": 0},
        {"RowId": 2, "ImageId": 1, "Feature": 1},
        {"RowId": 15, "ImageId": 6, "Feature": 14},
    ]


@pytest.fixture
def submission():
    return [{"RowId": 1, "Location": 32}, {"RowId": 1, "Location": 41}]


@pytest.fixture
def epoch_stats():
    return [EpochData() for i in range(2)]


@pytest.fixture
def train_dataset():
    return MockTrainDataloader()


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def test_format_file_name():
    return "some_format.csv"


def test_trainer(device, train_dataset):
    trainer = ModelTrainer(device, train_dataset, 0.8)
    assert trainer.device == device


def test_loop(device, train_dataset):
    trainer = ModelTrainer(device, train_dataset, 0.8)
    trainer.train_validate_loop(2)
    assert trainer.model is not None


def test_write_submisssion_exception(device, train_dataset, submission_format):
    trainer = ModelTrainer(device, train_dataset, 0.8)
    with pytest.raises(RuntimeError):
        trainer.predict_on_unlabeled_data(MockTestDataloader(), submission_format)


def test_predict_on_unlabeled_data(device, train_dataset, submission_format):
    trainer = ModelTrainer(device, train_dataset, 0.8)
    trainer.train_validate_loop(1)
    submission = trainer.predict_on_unlabeled_data(
        MockTestDataloader(), submission_format
    )
    assert len(submission) == len(submission_format)
    assert all("RowId" in submission_row for submission_row in submission)
    assert all("Location" in submission_row for submission_row in submission)
    assert all(row["Location"] is not None for row in submission)
    assert all(
        any(row["RowId"] == submission_row["RowId"] for submission_row in submission)
        for row in submission_format
    )


def test_save_losses(device, train_dataset):
    trainer = ModelTrainer(device, train_dataset, 0.8)
    trainer.train_validate_loop(2)
    with patch("builtins.open", mock_open()):
        trainer.save_losses("mock_file.csv")


def test_read_format_file(test_format_file_name, submission_format):
    mock_submission_format = """RowId,ImageId,FeatureName,Location
    1,1,left_eye_center_x
    2,1,left_eye_center_y
    15,6,left_eyebrow_outer_end_x"""

    with patch("builtins.open", mock_open(read_data=mock_submission_format)):
        result = Train.read_format_file(test_format_file_name)

    assert result == submission_format


def test_write_submission(device, submission):
    with patch("builtins.open", mock_open()):
        Train.write_submission(submission, "mock_name.csv")


def test_get_header():
    data = EpochData()
    data.get_csv_header()
