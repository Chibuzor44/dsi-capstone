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
        return util.metrics(y_test=y_true, y_predict=y_pred)


    def feature_importance(self, length = 13):
        """
        :param length: number of returned pair
        :return: list of tuples of terms and weight pair of positive and negative terms
        """
        terms = self.tfidf_vectorizer.get_feature_names()
        pos_terms, neg_terms = util.show_topics(self.lreg.coef_, terms, length = length)
        return pos_terms, neg_terms



def main(df, random_state, test_size, first_star, second_star, state = "Given Data"):
    """
    Runs the complete cycle of and builds a model. Takes a dataframe, 
    tokenizes, clean, vectorize and return model
    :param df: Pandas dataframe
    :param random_state: int
    :param test_size: int, split size for test
    :param first_star: int, first star rating
    :param second_star: int, second star rating
    :param state: string, state initials
    :return: logistic regression model object
    """
    if state:
        df = df[df["state"]==state]
    df = df[df.stars_rev.isin([first_star, second_star])]
    df = df[["text", "stars_rev"]].astype(str)
    
    df = util.to_binary(df, str(first_star), str(second_star))
    corpus = df["text"].values #list of reviews
    y = df["stars_rev"].values #target

    cleaned = util.clean_stem(corpus)
    
    X_train, X_test, y_train, \
            y_test = train_test_split(cleaned, y, test_size=test_size,
                                      random_state=random_state)

    model = ReviewClassifier() #instantiating model
    model.fit(X_train, y_train)
    
    #Estimating metrics
    matrix, recall, precision, accuracy = model.metrics_eval(X_test, y_test)
    print("Number of reviews: {}".format(len(corpus)), "\n")
    print("Confusion matrix")
    print(matrix, "\n")
    print("Recall: {}%".format(round(recall * 100, 2)))
    print("Precision: {}%".format(round(precision * 100, 2)))
    print("Accuracy: {}%".format(round(accuracy * 100, 2)))

    return model