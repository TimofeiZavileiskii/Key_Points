from dataclasses import dataclass, field, fields, asdict

import torch
import os
import numpy as np
from fontTools.misc import transform
from torch import nn
from torch.nn import BatchNorm1d
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import matplotlib.pyplot as plt
import csv
import logging

from Model import NeuralNetwork


def images_df_to_tensor(df):
    split_df = df["Image"].str.split(" ", expand=True)
    array = split_df.astype(np.float32).to_numpy()
    tensor = torch.tensor(array, dtype=torch.float32)
    tensor = tensor.reshape(-1, 1, 96, 96)
    return tensor


class FacialDataset(Dataset):
    def __init__(self, csv_path):
        self.images = pd.read_csv(csv_path)
        self.labels = self.images.drop("Image", axis=1)
        self.images = self.images[["Image"]]

        self.labels = (
            torch.tensor(self.labels.values, dtype=torch.float32).squeeze() / 95 - 0.5
        )
        self.images = images_df_to_tensor(self.images) / 255 - 0.5

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class UnlabeledFacialDataset(Dataset):
    def __init__(self, csv_path):
        self.images = pd.read_csv(csv_path)
        self.images = self.images[["Image"]]
        self.images = images_df_to_tensor(self.images) / 255 - 0.5

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, label, x):
        mask = ~torch.isnan(x)
        return self.mse_loss(x[mask], label[mask])


def create_model():
    return NeuralNetwork(3, 3)


def keep_keys(d, keys_to_keep):
    return {k: d[k] for k in keys_to_keep if k in d}


@dataclass
class EpochData:
    total_train_loss: float = 0.0
    average_train_loss: float = 0.0
    total_test_loss: float = 0.0
    average_test_loss: float = 0.0
    step_count: int = 0
    epoch: int = 0

    def get_csv_header(self):
        return [field.name for field in fields(EpochData)]


class ModelTrainer:
    def __init__(self, device, train_dataset, train_fraction):
        self.model = None
        self.device = device
        self.step_count = 0
        self.epoch_stats = []

        generator = torch.Generator().manual_seed(42)
        train_set, test_set = random_split(
            train_dataset, [train_fraction, 1 - train_fraction], generator=generator
        )

        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    def train_validate_loop(self, num_epochs):
        self.model = create_model().to(self.device)

        self.step_count = 0
        self.epoch_stats = []

        loss_fn = LossFunction()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        logging.info("Starting Training Loop")
        for t in range(num_epochs):
            curr_epoch = EpochData(0, 0, 0, 0, 0, 0)

            logging.info(f"Epoch {t} -------------------------------")
            self.train_epoch(loss_fn, optimizer, curr_epoch)
            self.test_epoch(loss_fn, curr_epoch)
            self.epoch_stats.append(curr_epoch)

        logging.info("Done!")

    def train_epoch(self, loss_fn, optimizer, epoch_stats):
        total_loss = 0
        total = 0

        self.model.train()
        for batch, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            total += y.size(0)

            # Backpropagation
            loss.backward()
            optimizer.step()
            self.step_count += 1
            optimizer.zero_grad()

        average_loss = total_loss / total
        logging.info(
            f"Average loss this epoch: {average_loss:>8f}, total train loss: {total_loss:>1f}, at step: {self.step_count}"
        )
        epoch_stats.average_train_loss = average_loss
        epoch_stats.total_train_loss = total_loss
        epoch_stats.step_count = self.step_count

    def test_epoch(self, loss_fn, epoch_stats):
        total_loss = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                total_loss += loss_fn(pred, y).item()
                total += y.size(0)

        average_loss = total_loss / total
        logging.info(
            f"Average test loss: {average_loss:>8f}, total test loss: {total_loss:>1f} \n"
        )

        epoch_stats.average_test_loss = average_loss
        epoch_stats.total_test_loss = total_loss

    def save_losses(self, filename):
        if len(self.epoch_stats) == 0:
            raise RuntimeError("No epoch stats saved")

        with open(filename, mode="w", newline="") as csvfile_out:
            csv_writer = csv.DictWriter(
                csvfile_out, fieldnames=self.epoch_stats[0].get_csv_header()
            )
            csv_writer.writeheader()

            for epoch_stats in self.epoch_stats:
                csv_writer.writerow(asdict(epoch_stats))

    def predict_on_unlabeled_data(self, dataloader, test_format):
        if not self.model:
            raise RuntimeError("No model has been trained yet")

        self.model.eval()  # Set the model to evaluation mode
        image_index_to_prediction = {}
        submission = []

        with torch.no_grad():
            for row in test_format:
                image_id = row["ImageId"]
                if image_id not in image_index_to_prediction:
                    model_prediction = self.model(
                        dataloader[image_id - 1].to(self.device).unsqueeze(0)
                    )
                    model_prediction = (model_prediction + 0.5) * 95
                    model_prediction = torch.clamp(model_prediction, min=0, max=95)
                    rounded_prediction = torch.round(model_prediction).to(torch.int)
                    image_index_to_prediction[image_id] = (
                        rounded_prediction.squeeze(0).cpu().numpy()
                    )

                row["Location"] = image_index_to_prediction[image_id][row["Feature"]]
                submission.append(keep_keys(row, ["Location", "RowId"]))

        return submission


def read_format_file(test_format_file_name):
    features_list = [
        "left_eye_center_x",
        "left_eye_center_y",
        "right_eye_center_x",
        "right_eye_center_y",
        "left_eye_inner_corner_x",
        "left_eye_inner_corner_y",
        "left_eye_outer_corner_x",
        "left_eye_outer_corner_y",
        "right_eye_inner_corner_x",
        "right_eye_inner_corner_y",
        "right_eye_outer_corner_x",
        "right_eye_outer_corner_y",
        "left_eyebrow_inner_end_x",
        "left_eyebrow_inner_end_y",
        "left_eyebrow_outer_end_x",
        "left_eyebrow_outer_end_y",
        "right_eyebrow_inner_end_x",
        "right_eyebrow_inner_end_y",
        "right_eyebrow_outer_end_x",
        "right_eyebrow_outer_end_y",
        "nose_tip_x",
        "nose_tip_y",
        "mouth_left_corner_x",
        "mouth_left_corner_y",
        "mouth_right_corner_x",
        "mouth_right_corner_y",
        "mouth_center_top_lip_x",
        "mouth_center_top_lip_y",
        "mouth_center_bottom_lip_x",
        "mouth_center_bottom_lip_y",
    ]

    map_feature_to_index = {feature: i for i, feature in enumerate(features_list)}

    with open(test_format_file_name, newline="") as csv_file:
        csv_reader = csv.DictReader(csv_file)

        format_dict = []
        for row in csv_reader:
            row["Feature"] = map_feature_to_index[row["FeatureName"]]
            row["ImageId"] = int(row["ImageId"])
            row["RowId"] = int(row["RowId"])
            del row["FeatureName"]
            del row["Location"]
            format_dict.append(row)

        return format_dict


def load_losses(filename: str):
    with open(filename, mode="r", newline="") as csvfile_in:
        reader = csv.DictReader(csvfile_in)
        str_fields = {f.name: f.type for f in fields(EpochData)}
        return [
            EpochData(**{key: str_fields[key](value) for key, value in row.items()})
            for row in reader
        ]


def dataclass_list_to_dict_of_lists(data) -> dict:
    if not data:
        return {}

    return {
        field.name: [getattr(item, field.name) for item in data]
        for field in fields(data[0])
    }


def plot_metrics(losses):
    lists_of_losses = dataclass_list_to_dict_of_lists(losses)

    fig, ax = plt.subplots(1, 1, figsize=(25, 10))

    # Plot loss
    ax.plot(
        lists_of_losses["average_train_loss"],
        label="Train Loss",
        marker="o",
        linestyle="-",
    )
    ax.plot(
        lists_of_losses["average_test_loss"],
        label="Test Loss",
        marker="s",
        linestyle="--",
    )

    ax.set_title("Loss Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()
    plt.show()


def write_submission(submission, submission_file_name):
    with open(submission_file_name, mode="w", newline="") as csvfile_out:
        writer = csv.DictWriter(csvfile_out, fieldnames=["RowId", "Location"])
        writer.writeheader()
        writer.writerows(submission)


def get_device():
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    logging.info(f"Device: {device}")
    return device


def plot_losses():
    plot_metrics(load_losses("training_stats.csv"))


def run_training():
    device = get_device()
    train_dataset = FacialDataset("data/training.csv")
    trainer = ModelTrainer(device, train_dataset, 0.8)

    trainer.train_validate_loop(1)
    trainer.save_losses("training_stats.csv")
    plot_metrics(trainer.epoch_stats)

    submission_loader = UnlabeledFacialDataset("data/test.csv")
    submission = trainer.predict_on_unlabeled_data(
        submission_loader, read_format_file("IdLookupTable.csv")
    )
    write_submission(submission, "submission.csv")


def count_parameters():
    model = create_model().eval()
    random_tensor = torch.randn(1, 1, 96, 96)
    _ = model(random_tensor)

    print(f"Model size {sum(p.numel() for p in model.parameters())}")
