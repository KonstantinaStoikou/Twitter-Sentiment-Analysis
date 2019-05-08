import statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


def dummy(doc):
    """ Dummy tokenizer to use when data are already tokenized. """
    return doc


def bag_of_words(series):
    """ Bag of words vectorization of tweets. Return a series of the vectors. """

    tweet_list = series.tolist()

    # if cleaned argument is true that means data are cleaned and already tokenized, else data are raw
    # if cleaned is True:
    bow_vectorizer = CountVectorizer(
        tokenizer=dummy, preprocessor=dummy, max_features=1000)
    # else:
    #     bow_vectorizer = CountVectorizer(
    #         max_df=1.0, min_df=1, max_features=2000, stop_words='english')

    matr = bow_vectorizer.fit_transform(tweet_list)

    ser = pd.Series(matr.toarray().tolist())
    # return series of vectors for tweets
    return ser


def tf_idf(series):
    """ Tf-Idf vectorization of tweets. Return a series of the vectors. """

    tweet_list = series.tolist()

    # if cleaned argument is true that means data are cleaned and already tokenized, else data are raw
    # if cleaned is True:
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=dummy, preprocessor=dummy, max_features=1000)
    # else:
    #     tfidf_vectorizer = TfidfVectorizer(
    #         max_df=1.0, min_df=1, max_features=1000, stop_words='english')

    matr = tfidf_vectorizer.fit_transform(tweet_list)

    ser = pd.Series(matr.toarray().tolist())
    # return series of vectors for tweets
    return ser


def train_model(series):
    """ Train a Word2Vec model on the given series of tokenized tweets. """

    tweet_list = series.tolist()

    model_w2v = Word2Vec(tweet_list, size=300, window=5,
                         min_count=2, sg=1, hs=0, negative=10, workers=2, seed=34)

    model_w2v.train(tweet_list, total_examples=len(tweet_list), epochs=20)

    return model_w2v


def tsne_plot(model, path):
    """ Create a TSNE model and plot it. """

    labels = []
    tokens = []
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2,
                      init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []

    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    # show only first 300 words
    for i in range(300):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(
            5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(path + 'pretrained_tsne_plot.png')
    plt.close()


def word_embedding(series, model_w2v):
    """ Word embedding vectorization of tweets. Return a series of the vectors. """

    tweet_list = series.tolist()
    # get number of features for vectors in model_w2v to use when creating arbitrary vector
    size = model_w2v.vector_size
    # for each word in a tweet find it's word embedding vector and append it to a list (of vectors)
    for i, tweet in enumerate(tweet_list):
        vec_list = []
        # if there is no word in the tweet, create a zero vector
        if len(tweet) == 0:
            vec = np.zeros(size)
            vec_list.append(vec)
        for j, word in enumerate(tweet):
            # if word is not in the trained model, create a random vector with values between min and max values of features of the previous word (if no previous word create a 0 vector)
            if word not in model_w2v.wv.vocab:
                if j - 1 >= 0:
                    min_val = min(vec_list[j-1])
                    max_val = max(vec_list[j-1])
                    vec = np.random.uniform(
                        low=min_val, high=max_val, size=size)
                else:
                    vec = np.zeros(size)
                vec_list.append(vec)
            else:
                vec_list.append(model_w2v[word])
        # convert vec_list to numpy array (an array of vectors)
        arr = np.array(vec_list)
        # change tweet_list item with mean of all the word vectors in the tweet
        tweet_list[i] = arr.mean(axis=0).tolist()

    ser = pd.Series(tweet_list)
    # return series of vectors for tweets
    return ser


def lexica_features(tweets, vectors, lex_list):
    """ Add extra features from lexicas to word embedding vectors. These features are: mean valence from each lexica, standard deviation of valences from each lexica, tweet length. So if , for example, there are 4 lexicas given, 4+4+1=9 new features wil be added in the vector.  """

    # a list with the tweet sentences
    tweet_list = tweets.tolist()
    # a list with the word embeddings for each tweet (make a deep copy of the list of lists)
    vectors_list = [x[:] for x in vectors.tolist()]
    # for each tweet find the average valence of it's words in each of the lexicas and append it to the word embedding vector
    for i, tweet in enumerate(tweet_list):
        # add feature to vector: length of tweet
        vectors_list[i].append(len(tweet))
        # if there is no word in the tweet, consider valence of every lexica as zero and add them to the word embedding vector as new features
        if len(tweet) == 0:
            for k in lex_list:
                # for mean
                vectors_list[i].append(0)
                # for standard deviation
                vectors_list[i].append(0)
        else:
            # for each given lexica find mean of valences of the words of the tweet
            for lex in lex_list:
                val_list = []
                for j, word in enumerate(tweet):
                    # if word is not in lexica, set valence as the mean of valences of the previous words (if no previous word set valence as zero)
                    if word not in lex:
                        if j - 1 >= 0:
                            num = sum(val_list) / len(val_list)
                        else:
                            num = 0
                        val_list.append(num)
                    else:
                        val_list.append(lex[word])
                # append to vector the mean of the word valences in this lexica
                mean = sum(val_list) / len(val_list)
                # add feature to vector: mean of valences from this lexica
                vectors_list[i].append(mean)
                # add feature to vector: standard deviation of valences from this lexica
                if len(val_list) > 1:
                    # standard deviation can be defined for at least two data points
                    sd = statistics.stdev(val_list)
                    vectors_list[i].append(sd)
                else:
                    vectors_list[i].append(0)

    ser = pd.Series(vectors_list)
    # return series of vectors for tweets
    return ser


def lexica_to_dict(lexica):
    """ Transform a lexica (text) to a dictionary (key: word, value: valence) """

    lexica_dict = {}
    with open(lexica) as f:
        for line in f:
            values = line.split()
            # check if there is a phrase and not just a word
            if len(values) > 2:
                phrase = ''
                for i in range(len(values) - 1):
                    phrase = phrase + ' ' + values[i]
                lexica_dict[phrase] = float(values[-1])
            else:
                word = values[0]
                lexica_dict[word] = float(values[1])

    return lexica_dict
