import pandas as pd
import gensim
from preprocessing import preprocess, stem, lemmatize
from analysis import analyze_data
from vectorization import bag_of_words, tf_idf, train_model, tsne_plot, word_embedding, lexica_features, lexica_to_dict
from gensim.models import Word2Vec
from classification import SVM, KNN


def main():
    """ The main function to invoke when the main.py program is running. """

    # example to csv
    # train_df.to_csv(
    #     "./../twitter_data/owncsv/cleaned_train.csv", index=False, header=True)

    # read train data and store them in a dataframe
    # train_df = pd.read_csv("./../twitter_data/train2017.tsv", engine='python',
    #                        sep='\t+', names=['Id', 'Number2', 'Sentiment', 'Tweet'])

    # DATA CLEANING
    # clean data and tokenize them
    # train_df = preprocess(train_df, 'Tweet')
    # lemmatize data
    # train_df['Tweet'] = train_df['Tweet'].apply(lambda x: stem(x))

    # save cleaned dataframes to pickle file
    # train_df.to_pickle("./pickles/stem_train.pkl")
    # load cleaned dataframes from pickle file
    train_df = pd.read_pickle("./pickles/lem_train.pkl")

    # DATA ANALYSIS
    # make statistical analysis on the data
    # analyze_data(train_df, 'Sentiment', 'Tweet')

    # TRAIN W2V MODEL
    # model_w2v = train_model(train_df['Tweet'])
    # model_w2v.save('pickles/model_w2v.pkl')
    # model_w2v = Word2Vec.load('pickles/model_w2v.pkl')
    # tsne_plot(model_w2v)

    # LOAD GOOGLE NEWS PRETRAINED MODEL
    model_w2v = gensim.models.KeyedVectors.load_word2vec_format(
        '../../GoogleNews-vectors-negative300.bin', binary=True)

    # CONVERT LEXICA TEXT FILES TO DICT
    affin_dict = lexica_to_dict('./../lexica/affin/affin.txt')
    emotweet_dict = lexica_to_dict('./../lexica/emotweet/valence_tweet.txt')
    generic_dict = lexica_to_dict('./../lexica/generic/generic.txt')
    nrc_dict = lexica_to_dict('./../lexica/nrc/val.txt')
    nrctag_dict = lexica_to_dict('./../lexica/nrctag/val.txt')

    # VECTORIZATION:
    bow_ser = bag_of_words(train_df['Tweet'])
    tfidf_ser = tf_idf(train_df['Tweet'])
    we_ser = word_embedding(train_df['Tweet'], model_w2v)
    lex_ser = lexica_features(train_df['Tweet'], we_ser, [affin_dict,                   emotweet_dict ,generic_dict, nrc_dict, nrctag_dict])

    # add new series to train dataframe
    train_df = pd.concat((train_df, bow_ser.rename('BOW')), axis=1)
    train_df = pd.concat((train_df, tfidf_ser.rename('TFIDF')), axis=1)
    train_df = pd.concat((train_df, we_ser.rename('WE')), axis=1)
    train_df = pd.concat((train_df, lex_ser.rename('LEX')), axis=1)

    # save cleaned dataframes to pickle file
    train_df[['Id', 'BOW', 'TFIDF', 'WE', 'LEX']].to_pickle("./pickles/vectors_train.pkl")
    # load cleaned dataframes from pickle file
    # train_df = pd.read_pickle("./pickles/vectors_train.pkl")

    # read test data to a pandas dataframe
    # test_df = pd.read_csv("./../twitter_data/test2017.tsv", engine='python',
    #                       sep='\t+', names=['Id', 'Number2', 'Sentiment', 'Tweet'])

    # DATA CLEANING
    # test_df = preprocess(test_df, 'Tweet')
    # test_df['Tweet'] = test_df['Tweet'].apply(lambda x: stem(x))

    # save cleaned dataframes to pickle file
    # test_df.to_pickle("./pickles/lem_test.pkl")
    # load cleaned dataframes from pickle file
    test_df = pd.read_pickle("./pickles/lem_test.pkl")

    # VECTORIZATION:
    bow_ser = bag_of_words(test_df['Tweet'])
    tfidf_ser = tf_idf(test_df['Tweet'])
    we_ser = word_embedding(test_df['Tweet'], model_w2v)
    lex_ser = lexica_features(test_df['Tweet'], we_ser, [affin_dict,                   emotweet_dict ,generic_dict, nrc_dict, nrctag_dict])

    # add new series to train dataframe
    test_df = pd.concat((test_df, bow_ser.rename('BOW')), axis=1)
    test_df = pd.concat((test_df, tfidf_ser.rename('TFIDF')), axis=1)
    test_df = pd.concat((test_df, we_ser.rename('WE')), axis=1)
    test_df = pd.concat((test_df, lex_ser.rename('LEX')), axis=1)

    # save new dataframes to pickle files
    test_df[['Id', 'BOW', 'TFIDF', 'WE', 'LEX']].to_pickle("./pickles/vectors_test.pkl")
    # load dataframes only with ids and tweet vectors from pickle files
    # test_df = pd.read_pickle("./pickles/vectors_test.pkl")

    # train_df[['WE', 'BOW', 'TFIDF']].head(1).to_csv(
    #     "./../twitter_data/owncsv/tessst.csv", index=False, header=True)
    # print(type(test_df['WE'].iloc[0]), type(
    #     test_df['BOW'].iloc[0]), type(test_df['TFIDF'].iloc[0]))

    # read answers file to use in the evaluation of predictions
    # eval_df = pd.read_fwf("./../twitter_data/SemEval2017_task4_subtaskA_test_english_gold.txt",
    #                       engine='python', sep='\t+', names=['Id', 'Sentiment'])

    # f1 = KNN(train_df['TFIDF'], train_df['Sentiment'],
    #          test_df['TFIDF'], eval_df['Sentiment'], 1)
    # print("tfidf + KNN f1 score is ", f1)


if __name__ == "__main__":
    main()
