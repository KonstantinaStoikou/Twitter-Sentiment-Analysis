import pandas as pd
import gensim
from preprocessing import preprocess, stem, lemmatize
from analysis import analyze_data
from vectorization import bag_of_words, tf_idf, train_model, tsne_plot, word_embedding, lexica_features, lexica_to_dict
from gensim.models import Word2Vec
from classification import SVM, KNN
from nltk import word_tokenize


def main():
    """ The main function to invoke when the main.py program is running. """

    # SET YOUR LOCAL PATHS to be used later in the program
    # paths to train/test/evaluation data files
    train_path = './../twitter_data/train2017.tsv'
    test_path = './../twitter_data/test2017.tsv'
    eval_path = './../twitter_data/SemEval2017_task4_subtaskA_test_english_gold.txt'
    # the path to the folder to store pickle files
    pickles_path = './pickles/'
    # the path to the pretrained word2vec model
    pretrainmodel_path = '../../GoogleNews-vectors-negative300.bin'
    # path to the folder with lexicas
    lexica_path = './../lexica/'
    # path to folder to store images
    images_path = './stats_images/'

    # TRAIN DATA PROCESSING
    # read train data and store them in a dataframe
    train_df = pd.read_csv(train_path, engine='python', sep='\t+',
                           names=['Id', 'Number2', 'Sentiment', 'Tweet'])

    # DATA CLEANING
    # clean data and tokenize them
    # train_df = preprocess(train_df, 'Tweet')
    # lemmatize data
    # train_df['Tweet'] = train_df['Tweet'].apply(lambda x: lemmatize(x))

    # train_df['Tweet'] = train_df['Tweet'].apply(lambda x: word_tokenize(x))

    # # save cleaned dataframes to pickle file
    # train_df.to_pickle(pickles_path + 'lem_train.pkl')
    # # load cleaned dataframes from pickle file
    train_df = pd.read_pickle(pickles_path + 'lem_train.pkl')

    # DATA ANALYSIS
    # make statistical analysis on the data
    # analyze_data(train_df, 'Sentiment', 'Tweet', images_path)

    # TRAIN W2V MODEL
    model_w2v = train_model(train_df['Tweet'])
    # model_w2v.save(pickles_path + 'model_w2v.pkl')
    # model_w2v = Word2Vec.load(pickles_path + 'model_w2v.pkl')
    # tsne_plot(model_w2v, images_path)

    # LOAD GOOGLE NEWS PRETRAINED MODEL
    # model_w2v = gensim.models.KeyedVectors.load_word2vec_format(
    #     pretrainmodel_path, binary=True)

    # CONVERT LEXICA TEXT FILES TO DICT
    affin_dict = lexica_to_dict(lexica_path + 'affin/affin.txt')
    emotweet_dict = lexica_to_dict(lexica_path + 'emotweet/valence_tweet.txt')
    generic_dict = lexica_to_dict(lexica_path + 'generic/generic.txt')
    nrc_dict = lexica_to_dict(lexica_path + 'nrc/val.txt')
    nrctag_dict = lexica_to_dict(lexica_path + 'nrctag/val.txt')

    # VECTORIZATION:
    # xbow_train = bag_of_words(train_df['Tweet'])
    # xtfidf_train = tf_idf(train_df['Tweet'])
    xwe_train = word_embedding(train_df['Tweet'], model_w2v)
    xlex_train = lexica_features(train_df['Tweet'], xwe_train, [
        affin_dict, emotweet_dict, generic_dict, nrc_dict, nrctag_dict])

    # save cleaned dataframes to pickle files
    # xbow_train.to_pickle(pickles_path + 'bow_train.pkl')
    # xtfidf_train.to_pickle(pickles_path + 'tfidf_train.pkl')
    # xwe_train.to_pickle(pickles_path + 'we_train.pkl')
    # xlex_train.to_pickle(pickles_path + 'lex_train.pkl')
    # load cleaned dataframes from pickle file
    # xbow_train = pd.read_pickle(pickles_path + 'bow_train.pkl')
    # xtfidf_train = pd.read_pickle(pickles_path + 'tfidf_train.pkl')
    # xwe_train = pd.read_pickle(pickles_path + 'we_train.pkl')
    # xlex_train = pd.read_pickle(pickles_path + 'lex_train.pkl')

    # TRAIN DATA PROCESSING
    # read test data to a pandas dataframe
    test_df = pd.read_csv(test_path, engine='python', sep='\t+',
                          names=['Id', 'Number2', 'Sentiment', 'Tweet'])

    # DATA CLEANING
    # test_df = preprocess(test_df, 'Tweet')
    # test_df['Tweet'] = test_df['Tweet'].apply(lambda x: lemmatize(x))

    # save cleaned dataframes to pickle file
    # test_df.to_pickle(pickles_path + 'lem_test.pkl')
    # load cleaned dataframes from pickle file
    test_df = pd.read_pickle(pickles_path + 'lem_test.pkl')

    # VECTORIZATION:
    # xbow_test = bag_of_words(test_df['Tweet'])
    # xtfidf_test = tf_idf(test_df['Tweet'])
    xwe_test = word_embedding(test_df['Tweet'], model_w2v)
    xlex_test = lexica_features(test_df['Tweet'], xwe_test, [
        affin_dict, emotweet_dict, generic_dict, nrc_dict, nrctag_dict])

    # # save new dataframes to pickle files
    # xbow_test.to_pickle(pickles_path + 'bow_test.pkl')
    # xtfidf_test.to_pickle(pickles_path + 'tfidf_test.pkl')
    # xwe_test.to_pickle(pickles_path + 'we_test.pkl')
    # xlex_test.to_pickle(pickles_path + 'lex_test.pkl')
    # # load dataframes only with ids and tweet vectors from pickle files
    # xbow_test = pd.read_pickle(pickles_path + 'bow_test.pkl')
    # xtfidf_test = pd.read_pickle(pickles_path + 'tfidf_test.pkl')
    # xwe_test = pd.read_pickle(pickles_path + 'we_test.pkl')
    # xlex_test = pd.read_pickle(pickles_path + 'lex_test.pkl')

    # read answers file to use in the evaluation of predictions
    eval_df = pd.read_fwf(eval_path, engine='python',
                          sep='\t+', names=['Id', 'Sentiment'])

    f1 = SVM(xwe_train, train_df['Sentiment'],
             xwe_test, eval_df['Sentiment'])
    print("we(trained) + lem + SVM f1 score is ", f1)

    f1 = SVM(xlex_train, train_df['Sentiment'],
             xlex_test, eval_df['Sentiment'])
    print("lex(trained) + lem + SVM f1 score is ", f1)


if __name__ == "__main__":
    main()
