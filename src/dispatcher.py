from sklearn.ensemble import ExtraTreesClassifier,\
     GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
# import xgboost as xgb
from xgboost import XGBClassifier

"""
For stacking you can use this way but a better way would be to generate
the outputs for all the classifiers individually and then stacking them
creating a class as it would be computationally less expensive."""
estimators = [('randomforest', RandomForestClassifier(n_estimators=200, n_jobs=-1,verbose=2)),
            ('extratrees', ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2)),
            ('gradientboostingclassifier', GradientBoostingClassifier(n_estimators=200,
     learning_rate=0.1, max_depth=3, verbose=2)),
     ("xgbm", XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, 
     verbosity=2))]




MODELS = {
    'randomforest': RandomForestClassifier(n_estimators=200, n_jobs=-1,
     verbose=2),
    'extratrees': ExtraTreesClassifier(n_estimators=200, n_jobs=-1,
     verbose=2),
    'gradientboostingclassifier': GradientBoostingClassifier(n_estimators=200,
     learning_rate=0.1, max_depth=3, verbose=2),
    'stackingclassifier': StackingClassifier(estimators=estimators,
     final_estimator=LogisticRegression()),
    "xgbm": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, 
     verbosity=2),
     "log_reg":LogisticRegression(C=0.123456789, solver='lbfgs',
      max_iter=1000, class_weight={0:1, 1:1.0349798013777922}, verbose=0, n_jobs=-1)

    
}