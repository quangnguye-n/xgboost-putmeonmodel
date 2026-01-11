import kaggle
from xgboost import XGBClassifier

kaggle.api.dataset_download_files('rounakbanik/the-movies-dataset', 
path='./movie_data', 
unzip=True)

