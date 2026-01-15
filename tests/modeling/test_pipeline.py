import re
import datetime
from unittest.mock import patch
from modeling.pipeline import get_timestamp, get_device


class TestGetTimestamp:
    def test_get_timestamp_format(self):
        timestamp = get_timestamp()
        pattern = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"

        assert re.match(pattern, timestamp)

    @patch("modeling.pipeline.datetime")
    def test_get_timestamp_fixed_value(self, mock_datetime_module):
        fixed_now = datetime.datetime(2023, 10, 27, 12, 30, 45)
        mock_datetime_module.datetime.now.return_value = fixed_now
        timestamp = get_timestamp()

        assert timestamp == "2023-10-27_12-30-45"


class TestGetDevice:
    @patch("modeling.pipeline.torch.cuda.is_available")
    def test_get_device_cuda(self, mock_cuda_available):
        mock_cuda_available.return_value = True
        device = get_device()

        assert device == "cuda"

    @patch("modeling.pipeline.torch.cuda.is_available")
    def test_get_device_cpu(self, mock_cuda_available):
        mock_cuda_available.return_value = False
        device = get_device()

        assert device == "cpu"
