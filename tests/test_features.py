import unittest
import numpy as np
import pandas as pd
from train_bike_demand import make_features, score

class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({
            "dteday": pd.date_range("2026-01-01", periods=20, freq="D"),
            "cnt": np.arange(20, dtype=float) * 10,
            "weekday": np.arange(20) % 7,
            "month": [1] * 20,
            "holiday": [0] * 20,
            "workingday": [1] * 20,
            "season": [1] * 20,
        })

    def test_target_day_is_not_a_feature(self):
        result = make_features(self.raw)
        self.assertNotIn("cnt", ["lag_2", "lag_3", "lag_7", "rolling_7"])
        first = result.iloc[0]
        self.assertEqual(first["lag_2"], self.raw.loc[6, "cnt"])

    def test_rolling_only_uses_completed_days(self):
        result = make_features(self.raw)
        self.assertAlmostEqual(
            result.iloc[0]["rolling_7"], self.raw.loc[0:6, "cnt"].mean()
        )

    def test_error_metrics(self):
        metrics = score([1, 2], [2, 2])
        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], 2**-0.5)

if __name__ == "__main__":
    unittest.main()
