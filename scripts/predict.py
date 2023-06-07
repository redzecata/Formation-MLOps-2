import os

import pandas as pd
from config import GENERATED_DATA_PATH, MODEL_PATH, PREDICTIONS_FOLDER

from formation_indus_ds_avancee.feature_engineering import prepare_features
from formation_indus_ds_avancee.train_and_predict import predict

generated_features_df = pd.read_csv(GENERATED_DATA_PATH, sep=';', nrows=10)
prepared_features_df = prepare_features(generated_features_df, training_mode=False)
predictions = predict(prepared_features_df, MODEL_PATH)
predictions.to_csv(os.path.join(PREDICTIONS_FOLDER, 'predictions.csv'), index=False)
print(predictions.head())
