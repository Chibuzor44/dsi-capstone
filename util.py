import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import nltk
from nltk import bigrams
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
nltk.download('wordnet')

def clean_stem(corpus):
    """
    This functions takes a corpus and return a list of tokenized,
    and stemmed documents with symbols and numbers stripped
    """
    sw = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer("[\w']+")
    lemma = WordNetLemmatizer()
    cleaned = [" ".join([lemma.lemmatize(word.lower()) for word in tokenizer.tokenize(doc)
            if regex(word) == False and word.lower() not in sw])
            for doc in corpus]
    return cleaned


def regex(word):
    """Checks if a string/word has digits"""
    checks = re.search(r"([0-9_])", word)
    return True if checks else False


def show_topics(vt, terms, length = 13):
    """
    This function prints out the topics and most associated words of the topic
    :param vt: V transpose matrix
    :param terms: list of vocabulary containing feature names
    :param length: number of words in a topic to be returned
    :return: prints topics and list of terms
    """
    for i, beta in enumerate(vt, 1):
        pos_sort = sorted(zip(terms, beta), key=lambda x: x[1], reverse=True)[:length]
        neg_sort = sorted(zip(terms, beta), key=lambda x: x[1], reverse=False)[:length]

        pos_term = sorted({k:v for k,v in pos_sort}.items(),
                          key=lambda x: x[1], reverse=True)
        neg_term = sorted({k:v for k,v in neg_sort}.items(),
                          key=lambda x: x[1], reverse=False)

    return pos_term, neg_term


def metrics(df=None,y_test=None, y_predict=None):
    """
    This function calculates evaluation metrics for two arrays of true
    and predicted values or from spark dataframe with tp,tn,fp,fn calculated.
    ie matrix, recall precision, and accuracy.
    :param df: spark dataframe
    :param y_test: array of true prediction
    :param y_predict: array of predicted values
    :return: confusion matrix, recall, precision,  and accuracy
    """
    if df:
        tp, tn, fp, fn = df.collect()[0]
    else:
        tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    matrix = np.array([[tp, fp], [fn, tn]])
    recall = tp/(tp+fn)
    precision = tp/(fp+tp)
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    return matrix, recall, precision, accuracy




def service_section(corpus, terms):
    """
    This function returns sections of service an list of words used to 
    describe these sections
    :param corpus: list of reviews
    :param terms: descriptive adjectives
    :return: dataframe, list of tuples of words and list of
            words describing them
    """
    term = [word[0] for word in terms]
    dic = defaultdict(list)
    desc = np.array(term)
    for doc in corpus:
        grams = bigrams(doc.split())
        for tup in list(grams):
            if tup[0] in term:
                indx = term.index(tup[0])
                if tup[1] in dic and len(tup[1]) > 3:
                    dic[tup[1]].append(indx)
                else:
                    dic[tup[1]] = [indx]

    with open("../break_week/aspect.pickle", "rb") as f:
        aspect = pickle.load(f)

    asp = {"Food":[], "Service":[], "Location":[], "Drink":[], "Cost":[]}
    for k,v in dic.items():
        for key, val in aspect.items():
            if k in val:
                asp[key].extend(v)

    lst = sort({k: len(desc[[v]]) for k,v in asp.items()})
    lst_tup = sorted({k: dict(Counter(desc[[v]])) for k,v in asp.items()}.items(),
                     key=lambda x: len(x[1]), reverse=True)
    df_service = pd.DataFrame(lst, columns=["Aspect", "Level of experience"])
    return df_service, lst_tup


def sort(dic):
    """Returns a sorted list of tuples of key and value pair"""
    return sorted(dic.items(), key=lambda x: x[1], reverse=True)

def to_binary(df, first_star, second_star):
    """
    Returns a dataframe with stars_rev col being 0s and 1s
    :param df: dataframe
    :param first_star: int (star rating)
    :param second_star: int (star rating)
    :return: dataframe
    """
    label = {"stars_rev": {first_star: 0, second_star: 1}}
    df.replace(label, inplace=True)
    return df



def vary_ratings(model, df, first_star, second_star, indx=1, bus_name=None):
    """
    Prints a list of tuples of key and value pair negative and positive
    aspects, bar charts of each, and confusion matrix. The key is the aspect and value
    is a dictionary words describing the aspect and the count of the words.
    :param model: trained logistic regression model object
    :param df: dataframe
    :param first_star: star rating
    :param second_star: star rating
    :param indx: int
    :param bus_name: Name of the restaurant
    :return:
    """
    unique_id = pd.unique(df["business_id"])
    if bus_name:
        df_rest = df[["bus_name","text", "stars_rev"]][df["bus_name"]==bus_name]
    else:
        df_rest = df[["bus_name","text", "stars_rev"]][df["business_id"]==unique_id[indx]]
    
    print("Restaurant Name: {} \n".format(df_rest["bus_name"].iloc[0]))
    
    df_rest = df_rest[df_rest.stars_rev.isin([first_star, second_star])]
    df_rest = df_rest[["text", "stars_rev"]].astype(str)
    df_rest = to_binary(df_rest, str(first_star), str(second_star))
    y_rest = df_rest["stars_rev"].values
    rest_corpus = clean_stem(df_rest["text"].values)
    print("Size of corpus: {}".format(len(rest_corpus)), "\n")
    pos_term, neg_term = model.feature_importance(15)
    matrix, recall, precision, accuracy = model.metrics_eval(rest_corpus, y_rest)
    print("Recall: {}%".format(round(recall, 2)))
    print("Precision: {}%".format(round(precision, 2)))
    print("Accuracy: {}%".format(round(accuracy*100, 2)))
    df_neg, lst_neg = service_section(rest_corpus, neg_term)
    df_pos, lst_pos = service_section(rest_corpus, pos_term)
    print("\n",lst_neg, "\n")
    print("\n", lst_pos, "\n")
    sns.barplot(x="Aspect", y="Level of experience", data=df_neg)
    plt.title("Negative experience")
    plt.show()
    plt.title("Positive experience")
    sns.barplot(x="Aspect", y="Level of experience", data=df_pos)
    plt.show()


def display(model, df, first_star, second_star, state=None, bus_name=None):
    """
    Returns positive and negative aspect dataframe, restaurant name, number of reviews,
    Recall, Precision, Accuracy, list of tuples of aspects as key and value as dict of
    words describing them.
    :param model:
    :param df:
    :param first_star:
    :param second_star:
    :param state:
    :param bus_name:
    :return:
    """

    df_rest = df[["bus_name", "text", "stars_rev", "state"]]
    df_rest = df_rest[(df_rest["state"] == state) & (df_rest["bus_name"] == bus_name)]

    name = "Restaurant Name: " + bus_name

    df_rest = df_rest[df_rest.stars_rev.isin([first_star, second_star])]
    df_rest = df_rest[["text", "stars_rev"]].astype(str)
    df_rest = to_binary(df_rest, str(first_star), str(second_star))
    y_rest = df_rest["stars_rev"].values
    rest_corpus = clean_stem(df_rest["text"].values)

    size = "Size of corpus: {}".format(len(rest_corpus))
    pos_term, neg_term = model.feature_importance(15)
    matrix, recall, precision, accuracy = model.metrics_eval(rest_corpus, y_rest)
    Recall = "Recall: {}%".format(round(recall, 2))
    Precision = "Precision: {}%".format(round(precision, 2))
    Accuracy = "Accuracy: {}%".format(round(accuracy * 100, 2))
    df_neg, lst_neg = service_section(rest_corpus, neg_term)
    df_pos, lst_pos = service_section(rest_corpus, pos_term)
    return df_pos, df_neg, name, size, Recall, Precision, Accuracy, lst_neg[:15], lst_pos[:15]


