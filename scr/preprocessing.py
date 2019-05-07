import re
from bs4 import BeautifulSoup
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer


def preprocess(df, columns):
    """ Preprocess given dataframe by cleaning, tokenizing and stemming
    data (df: dataframe, columns: columns name to clean data). """

    # make a copy of the original dataset so that it remains unmodified
    df_copy = df.copy()
    # clean data of the given columns of the dataframe
    df_copy[columns] = df_copy[columns].apply(lambda x: clean(x))
    # tokenize data and remove stopwords
    df_copy[columns] = df_copy[columns].apply(lambda x: tokenize(x))
    # return cleaned dataframe
    return df_copy


def stem(tokens):
    """ Stem all words in given list of tokens. """

    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(token) for token in tokens]

    return stems


def lemmatize(tokens):
    """ Lemmatize all words in given list of tokens. """

    lemmatizer = WordNetLemmatizer()
    lems = [lemmatizer.lemmatize(token, get_wordnet_pos(token))
            for token in tokens]

    return lems


def get_wordnet_pos(word):
    """ Map POS tag to first character lemmatize() accepts. """
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def clean(text):
    """ Clean the text and return the cleaned text. """

    # make all letters lowercase
    text = text.lower()
    # remove @mentions
    text = re.sub('@[A-Za-z0-9]+', '', text)
    # remove URLs
    text = re.sub('https?://[A-Za-z0-9./]+', '', text)
    # remove HTML encoding
    text = BeautifulSoup(text, 'lxml').get_text()
    # remove punctuation, numbers, hashtag symbols
    text = re.sub("[^a-zA-Z]", ' ', text)
    # remove extra whitespaces between words
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    """ Tokenize text into words and remove stopwords, return filtered tokens list. """

    tokens = word_tokenize(text)
    # remove stopwords from tokens
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    return filtered
