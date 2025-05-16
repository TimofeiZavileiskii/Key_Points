from Model import *
import torch
from torch import nn
import pytest


@pytest.fixture
def random_batch():
    return torch.randn(16, 1, 96, 96)


@pytest.fixture
def input_parameters():
    return 100


@pytest.fixture
def output_parameters():
    return 200


@pytest.fixture
def random_1d_batch(input_parameters):
    return torch.randn(16, input_parameters)


@pytest.fixture
def random_2d_batch(input_parameters):
    return torch.randn(16, input_parameters, 96, 96)


def test_model(random_batch):
    model = NeuralNetwork().eval()
    result = model(random_batch)

    assert result.shape == (16, 30)


def test_linear_block(random_1d_batch, input_parameters, output_parameters):
    block = LinearBlock(input_parameters, output_parameters)
    output = block(random_1d_batch)
    assert output.shape == (16, output_parameters)


def test_conv_block(random_2d_batch, input_parameters, output_parameters):
    block = ConvBlock(input_parameters, output_parameters)
    output = block(random_2d_batch)
    assert output.shape[0:2] == (16, output_parameters)


def test_conv_res_block(random_2d_batch, input_parameters, output_parameters):
    block = ConvResBlock(input_parameters, output_parameters, 3)
    output = block(random_2d_batch)
    assert output.shape[0:2] == (16, output_parameters)


def test_linear_res_block(random_1d_batch, input_parameters, output_parameters):
    block = LinearResBlock(input_parameters, output_parameters, 3)
    output = block(random_1d_batch)
    assert output.shape == (16, output_parameters)
