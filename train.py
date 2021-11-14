import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb

df = pd.read_csv("Downloads/heart.csv")
categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
numerical = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

df_train = df[categorical+numerical]
y_train = df["HeartDisease"].values

dv = DictVectorizer(sparse=False)
train_dict = df_train[categorical+numerical].to_dict(orient="records")
X_train = dv.fit_transform(train_dict)

xgb_params = {
            'eta': 0.02,
            'max_depth': 3,
            'min_child_weight': 1,

            'objective': 'binary:logistic',
            'eval_metric': 'auc',

            'nthread': 8,
            'seed': 1,
            'verbosity': 1,
        }

features = dv.get_feature_names()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

model = xgb.train(xgb_params, dtrain, num_boost_round=200, verbose_eval=5)