import numpy as np
from sklearn.metrics import confusion_matrix


def clean_stem(corpus, tokenizer, lemma, sw):
    """
    This functions takes a corpus and return a list of tokenized,
    and stemmed documents with symbols and numbers stripped
    """
    cleaned = [[lemma.lemmatize(word.lower()) for word in tokenizer.tokenize(doc)
            if word.isdigit() == False and word not in sw]
            for doc in corpus]
    return cleaned



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



if __name__ == "__main__":
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    sw = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer("[\w']+")
    st = PorterStemmer()

    cup = ["Deserves a zero. I was highly disappointed. We went for my husband's birthday expecting a meal worthy of the $200 we spent.  The food was very bland. I was so upset I didn't even bother to order dessert.  I thought if dessert tastes anything like dinner, I'm not going to bother. Noone bothered to ask how the food was. The chef didn't bother to come to our table,  even though he spoke to individuals at the three other tables surrounding us. Thank God I opted for the indoor gondola ride after dinner which helped to improve my mood for the evening.  I won't be back here or to any of his restaurants. I'm a good person so I still left a very generous tip;  after all it wasn't the waiter's fault the food was just ok.",
    "This was by far the worst meal I have ever had.  Not steak,   Meal!  I went with a group of 4, only 1 steak was cooked correctly and it was super fatty. I had a filet ordered medium to medium well black, it showed up well done with no char whatsoever.  My friend ordered hers blue and it showed up rare. My husband's was fine but like I said super fatty. The other friends was over cooked as well along with the snottiest looking disgusting foie gras I have ever seen. Bottom line_ you go to Vegas to have a great meal and it sucked.  All of it sucked. Go somewhere else for $100pp."]
    print(clean_stem(cup, tokenizer, st, sw))