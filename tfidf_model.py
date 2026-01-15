from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from config import TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_STOP_WORDS, TFIDF_NGRAM_RANGE

# create tf-idf vectoriser
def fit_tfidf(text_series):
    vectoriser = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, # limits vocab with 500 most important words since we are using small dataset
        min_df=TFIDF_MIN_DF, # word has to appear in at least two movies
        stop_words=TFIDF_STOP_WORDS, # remove common words like 'the', 'a', 'is'
        ngram_range=TFIDF_NGRAM_RANGE
    )

    # convert text to tf-idf vectors
    X = vectoriser.fit_transform(text_series)
    return vectoriser, X