from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier
from sbi.inference import SNPE


ESTIMATORS = {
    # regression/prediction algorithms
    'gb_r': lambda **kwargs: XGBRegressor(**kwargs),
    'rf': lambda **kwargs: RandomForestRegressor(**kwargs),
    'mlp_r': lambda **kwargs: MLPRegressor(**kwargs),

    # posterior estimators
    'snpe': lambda **kwargs: SNPE(**kwargs),

    # probabilistic classification algorithms (for likelihood estimation)
    'qda': lambda **kwargs: QuadraticDiscriminantAnalysis(**kwargs),
    'mlp_c': lambda **kwargs: MLPClassifier(**kwargs),
    'gb_c': lambda **kwargs: XGBClassifier(**kwargs),
}
