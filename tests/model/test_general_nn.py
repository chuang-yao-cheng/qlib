import unittest
from unittest.mock import patch

import torch

from qlib.tests import TestAutoData


class TestNN(TestAutoData):
    def test_both_dataset(self):
        try:
            from qlib.contrib.model.pytorch_general_nn import GeneralPTNN
            from qlib.data.dataset import DatasetH, TSDatasetH
            from qlib.data.dataset.handler import DataHandlerLP
        except ImportError:
            print("Import error.")
            return

        data_handler_config = {
            "start_time": "2008-01-01",
            "end_time": "2020-08-01",
            "instruments": "csi300",
            "data_loader": {
                "class": "QlibDataLoader",  # Assuming QlibDataLoader is a string reference to the class
                "kwargs": {
                    "config": {
                        "feature": [["$high", "$close", "$low"], ["H", "C", "L"]],
                        "label": [["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]],
                    },
                    "freq": "day",
                },
            },
            # TODO: processors
            "learn_processors": [
                {
                    "class": "DropnaLabel",
                },
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ],
        }
        segments = {
            "train": ["2008-01-01", "2014-12-31"],
            "valid": ["2015-01-01", "2016-12-31"],
            "test": ["2017-01-01", "2020-08-01"],
        }
        data_handler = DataHandlerLP(**data_handler_config)

        # time-series dataset
        tsds = TSDatasetH(handler=data_handler, segments=segments)

        # tabular dataset
        tbds = DatasetH(handler=data_handler, segments=segments)

        model_l = [
            GeneralPTNN(
                n_epochs=2,
                batch_size=32,
                n_jobs=0,
                pt_model_uri="qlib.contrib.model.pytorch_gru_ts.GRUModel",
                pt_model_kwargs={
                    "d_feat": 3,
                    "hidden_size": 8,
                    "num_layers": 1,
                    "dropout": 0.0,
                },
            ),
            GeneralPTNN(
                n_epochs=2,
                batch_size=32,
                n_jobs=0,
                pt_model_uri="qlib.contrib.model.pytorch_nn.Net",  # it is a MLP
                pt_model_kwargs={
                    "input_dim": 3,
                },
            ),
        ]

        for ds, model in list(zip((tsds, tbds), model_l)):
            model.fit(ds)  # It works
            model.predict(ds)  # It works


class TestGeneralPTNNDeviceSelection(unittest.TestCase):
    def test_explicit_cpu_device(self):
        from qlib.contrib.model.pytorch_general_nn import (
            GeneralPTNN,
            _resolve_torch_device,
        )

        self.assertEqual(_resolve_torch_device("cpu", 0), torch.device("cpu"))
        model = GeneralPTNN(
            n_epochs=1,
            batch_size=8,
            n_jobs=0,
            device="cpu",
            pt_model_uri="qlib.contrib.model.pytorch_nn.Net",
            pt_model_kwargs={"input_dim": 3},
        )
        self.assertEqual(model.device, torch.device("cpu"))

    def test_auto_device_prefers_mps_after_cuda(self):
        from qlib.contrib.model import pytorch_general_nn

        with patch.object(
            pytorch_general_nn.torch.cuda, "is_available", return_value=False
        ), patch.object(
            pytorch_general_nn.torch.backends.mps, "is_available", return_value=True
        ):
            device = pytorch_general_nn._resolve_torch_device("auto", 0)

        self.assertEqual(device, torch.device("mps"))

    def test_explicit_mps_device_fails_when_unavailable(self):
        from qlib.contrib.model import pytorch_general_nn

        with patch.object(
            pytorch_general_nn.torch.backends.mps, "is_available", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "mps"):
                pytorch_general_nn._resolve_torch_device("mps", 0)


if __name__ == "__main__":
    unittest.main()
