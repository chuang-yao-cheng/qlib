# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

from qlib.contrib.model import all_model_classes
import qlib.contrib.model as model_pkg


class TestAllFlow(unittest.TestCase):
    def test_0_initialize(self):
        names = [model_class.__name__ for model_class in all_model_classes]
        self.assertIn("LGBModel", names)
        self.assertIn("DNNModelPytorch", names)
        self.assertIsNotNone(model_pkg.LGBModel)


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllFlow("test_0_initialize"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
