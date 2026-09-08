import xgboost

HORIZONS = [15, 30, 60, 90, 120]
NATIVE_STEP_MIN = 5


# ─────────────────────────── Zero-parameter baseline ───────────────────────────

class PersistencePredictor:
    def __init__(self, num_horizons: int = len(HORIZONS)):
        self.num_horizons = num_horizons

    def fit(self, X, y=None) -> 'PersistencePredictor':
        return self  # nothing to fit

    def predict(self, X):
        import numpy as np
        return np.repeat(np.asarray(X)[:, -1:], self.num_horizons, axis=1)


# ─────────────────────────── Classical, non-recurrent baselines ───────────────────────────

class LinearRegressionPredictor:
    def __init__(self, num_horizons: int = len(HORIZONS)):
        from sklearn.linear_model import LinearRegression
        self.num_horizons = num_horizons
        self.models = [LinearRegression() for _ in range(num_horizons)]

    def fit(self, X, y) -> 'LinearRegressionPredictor':
        # X: [N, T] flattened CGM window, y: [N, num_horizons]
        for h in range(self.num_horizons):
            self.models[h].fit(X, y[:, h])
        return self

    def predict(self, X):
        import numpy as np
        return np.stack([m.predict(X) for m in self.models], axis=1)


class XGBoostPredictor:
    def __init__(self, random_state: int, num_horizons: int = len(HORIZONS), **xgb_kwargs):
        from xgboost import XGBRegressor
        self.num_horizons = num_horizons
        params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, random_state=random_state)
        params.update(xgb_kwargs)
        self.models = [XGBRegressor(**params) for _ in range(num_horizons)]

    def fit(self, X, y, eval_set=None) -> 'XGBoostPredictor':
        for h in range(self.num_horizons):
            es = [(eval_set[0], eval_set[1][:, h])] if eval_set is not None else None
            self.models[h].fit(X, y[:, h], eval_set=es, verbose=False)
        return self

    def predict(self, X):
        import numpy as np
        return np.stack([m.predict(X) for m in self.models], axis=1)