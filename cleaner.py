

def clean_stem(corpus, tokenizer, st, sw):
    """
    This functions takes a corpus and return a list of tokenized,
    and stemmed documents with symbols and numbers stripped
    """
    cleaned = [" ".join([st.stem(word) for word in tokenizer.tokenize(doc)
            if word.isdigit() == False and word not in sw])
            for doc in corpus]
    return cleaned


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
