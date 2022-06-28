from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


MODELS = {
    'randomforest': RandomForestClassifier(n_estimators=200, n_jobs=-1,
    verbose=2),
    'extratrees': ExtraTreesClassifier(n_estimators=200, n_jobs=-1,
    verbose=2)
}