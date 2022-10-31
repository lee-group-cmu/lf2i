from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor

from sbi.inference import SNPE


ESTIMATORS = {
    # regression/prediction algorithms
    'gb': lambda **kwargs: XGBRegressor(**kwargs),
    'rf': lambda **kwargs: RandomForestRegressor(**kwargs),
    'mlp': lambda **kwargs: MLPRegressor(**kwargs),

    # posterior estimators
    'snpe': lambda **kwargs: SNPE(**kwargs)

    # probabilistic classification algorithms (for likelihood estimation)

}
