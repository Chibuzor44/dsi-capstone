import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import util



class ReviewClassifier:
    """
    Restaurant review classifier of star-ratings using customer review text.
    """
    def __init__(self):
        
        self.X_train, self.y_train = None, None
        self.lreg = LogisticRegression()
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                                stop_words="english")


    def fit(self, X_train, y_train):
        """
        This methods trains the model
        :param X_train: 1-dimensional array of review texts
        :param y_train: 1-dimensional array of target values
        :return: trained model
        """
        self.tfidf_vectorizer.fit_transform(X_train)
        self.X_train = self.tfidf_vectorizer.transform(X_train)
        self.y_train = y_train
        self.lreg.fit(self.X_train, self.y_train)
        return self


    def predict(self, X_test):
        """
        :param X_test: 1-dimensional array of review texts
        :return: 1-dimensional array of predicted values
        """
        X_test = self.tfidf_vectorizer.transform(X_test)
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


    def feature_importance(self, length = 13):
        """
        :param length: number of returned pair
        :return: list of tuples of terms and weight pair of positive and negative terms
        """
        terms = self.tfidf_vectorizer.get_feature_names()
        pos_terms, neg_terms = util.show_topics(self.lreg.coef_, terms, length = length)
        return pos_terms, neg_terms




# def vectorizer(corpus, max_df=0.95, min_df=2, stop_words='english'):
#     """
#     Converts corpus to term frequency inverse document frequency matrix
#     :param corpus: list of strings of documents
#     :param max_df: ignore terms that have a document frequency strictly higher than the given threshold
#     :param min_df: ignore terms that have a document frequency strictly lower than the given threshold
#     :param stop_words: check_stop_list and the appropriate stop list is returned
#     :return: sparse matrix and 1-D array of vocabulary vector
#     """
#     tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
#                                        stop_words=stop_words)

#     tfidf = tfidf_vectorizer.fit_transform(corpus)
#     terms = tfidf_vectorizer.get_feature_names()
#     X = tfidf.toarray()
#     return X, terms


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



def main(df, tokenizer, lemma, sw, random_state, test_size, length=15, state = "Given Data"):

    corpus = df["text"].values #list of reviews
    y = df["stars_rev"].values #target

    cleaned = util.clean_stem(corpus, tokenizer, lemma, sw)
    
    X_train, X_test, y_train, \
            y_test = train_test_split(cleaned, y, test_size=test_size,
                                      random_state=random_state)

    model = ReviewClassifier() #instantiating model
    model.fit(X_train, y_train)
    
    #Estimating metrics
    matrix, recall, precision, accuracy = model.metrics_eval(X_test, y_test)
    print("Results for " + state, "\n","\n","\n")
    print("Number of reviews: {}".format(len(corpus)), "\n")
    print("Confusion matrix")
    print(matrix, "\n")
    print("Recall: {}%".format(round(recall * 100, 2)))
    print("Precision: {}%".format(round(precision * 100, 2)))
    print("Accuracy: {}%".format(round(accuracy * 100, 2)))
    print("\n","\n","\n")

    pos_terms, neg_terms = model.feature_importance(length)
    print("Words customers who gave 5-star reviews used to describe their experience")
    print("-------------------------------------------------------------------------\n")
    print(pos_terms)
    print("\n Words customers who gave 1-star reviews used to describe their experience")
    print("-------------------------------------------------------------------------\n")
    print(neg_terms)
