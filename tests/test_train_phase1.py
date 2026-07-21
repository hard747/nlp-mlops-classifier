from unittest.mock import patch

from transformers import TrainingArguments

from train.common import detect_device


def test_detect_device_returns_cpu_without_cuda():
    with patch("train.common.torch.cuda.is_available", return_value=False):
        assert detect_device() == "cpu"


def test_detect_device_returns_cuda_when_available():
    with patch("train.common.torch.cuda.is_available", return_value=True):
        assert detect_device() == "cuda"


def test_training_arguments_construct_smoke():
    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        fp16=False,
        report_to="none",
    )
    assert args.per_device_train_batch_size == 4
    assert args.num_train_epochs == 1
