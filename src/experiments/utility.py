import warnings
from typing import FrozenSet

import numpy as np
from pydvl.utils.utility import Utility

__all__ = ["FeatureValuationUtility"]


class FeatureValuationUtility(Utility):
    def _utility(self, indices: FrozenSet) -> float:
        if not indices:
            return 0.0

        x_train, y_train = self.data.get_training_data(list(indices))
        x_test, y_test = self.data.get_test_data(list(indices))
        try:
            self.model.fit(x_train, y_train)
            score = float(self.scorer(self.model, x_test, y_test))
            # Some scorers raise exceptions if they return NaNs, some might not
            if np.isnan(score):
                if self.show_warnings:
                    warnings.warn(f"Scorer returned NaN", RuntimeWarning)
                return self.default_score
            return score
        except Exception as e:
            if self.catch_errors:
                if self.show_warnings:
                    warnings.warn(str(e), RuntimeWarning)
                return self.default_score
            raise e
