import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import cleaner



class ReviewClassifier:
    """
    Restaurant review classifier of star-ratings using customer review text.
    """
    def __init__(self):
        
        self.X_train, self.y_train = None, None
        self.lreg = LogisticRegression()



    def fit(self, X_train, y_train):
        """
        This methods trains the model
        :param X_train: 2-dimensional array of sparse matrix
        :param y_train: 1-dimensional array of target values
        :return: trained model
        """
        self.X_train, self.y_train = X_train, y_train
        self.lreg.fit(self.X_train, self.y_train)
        return self


    def predict(self, X_test):
        """
        :param X_test: 2-dimensional array of sparse matrix
        :return: 1-dimensional array of predicted values
        """
        return self.lreg.predict(X_test)

    def metrics_eval(self, X_test, y_true):
        """
        Calculates the confusion matrix, recall, precision, accuracy
        of model predictions.
        :param X_test: 2-dimensional array of sparse matrix
        :param y_true: 1-dimensional array of true target values
        :return: 2-D arrays confusion matrix, floats of recall, precision and accuracy
        """
        y_pred = self.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        matrix = np.array([[tp, fp], [fn, tn]])
        recall = tp / (tp + fn)
        precision = tp / (fp + tp)
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        return matrix, recall, precision, accuracy


    def feature_importance(self, terms, length = 13):
        """
        :param terms: model vocabulary vector (array of word features)
        :param length: number of returned pair
        :return: list of tuples of terms and weight pair of positive and negative terms
        """
        pos_terms, neg_terms = cleaner.show_topics(self.lreg.coef_, terms, length = length)
        return pos_terms, neg_terms




def vectorizer(corpus, max_df=0.95, min_df=2, stop_words='english'):
    """
    Converts corpus to term frequency inverse document frequency matrix
    :param corpus: list of strings of documents
    :param max_df: ignore terms that have a document frequency strictly higher than the given threshold
    :param min_df: ignore terms that have a document frequency strictly lower than the given threshold
    :param stop_words: check_stop_list and the appropriate stop list is returned
    :return: sparse matrix and 1-D array of vocabulary vector
    """
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                                       stop_words=stop_words)

    tfidf = tfidf_vectorizer.fit_transform(corpus)
    terms = tfidf_vectorizer.get_feature_names()
    X = tfidf.toarray()
    return X, terms


def train_test_splits(X, y, test_size, random_state=0):
    """
    :param X: 2-dimensional feature matrix
    :param y: 1-dimensional target array
    :param test_size: ratio of data testset size
    :param random_state: random seed
    :return: 2-dimensional array of X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, \
            y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test



# def main():
