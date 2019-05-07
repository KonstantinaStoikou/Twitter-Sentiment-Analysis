import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib.ticker import MaxNLocator


def analyze_data(df, sentiment_col, tweet_col):
    """ Make statistical analysis on data and visualize it. """

    # create empty dictionaries to store all encountered words and their frequencies
    all_dict = {}
    pos_dict = {}
    neg_dict = {}
    neu_dict = {}
    # initialize counters to counter total number of tweets based on their emotion
    pos_count = 0
    neg_count = 0
    neu_count = 0

    # iterate through each row of the df
    for index, row in df.iterrows():
        if row[sentiment_col] == "positive":
            pos_count = iterate_words(
                pos_count, row[tweet_col], all_dict, pos_dict)

        if row[sentiment_col] == "negative":
            neg_count = iterate_words(
                neg_count, row[tweet_col], all_dict, neg_dict)

        if row[sentiment_col] == "neutral":
            neu_count = iterate_words(
                neu_count, row[tweet_col], all_dict, neu_dict)

    # visualize statistics
    visualize_stats(all_dict, 'all_plot.png', 'all_cloud.png',
                    'Word frequency in all tweets')
    visualize_stats(pos_dict, 'pos_plot.png', 'pos_cloud.png',
                    'Word frequency in positive tweets')
    visualize_stats(neg_dict, 'neg_plot.png', 'neg_cloud.png',
                    'Word frequency in negative tweets')
    visualize_stats(neu_dict, 'neu_plot.png', 'neu_cloud.png',
                    'Word frequency in neutral tweets')

    # make plot for emotion frequency
    emotions = ('Positive', 'Negative', 'Neutral')
    freq = [pos_count, neg_count, neu_count]
    sns.set_style("darkgrid")
    ax = plt.figure().gca()
    ax.xaxis.grid(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.bar(range(len(emotions)), freq, align='center',
            color=['forestgreen', 'firebrick', 'goldenrod'])
    plt.xticks(range(len(emotions)), emotions)
    plt.title('Tweet frequency based on emotion')
    plt.savefig('stats_images/emotion_plot.png')
    plt.close()

    # make pie for emotion frequency
    sizes = [pos_count / len(df.index), neg_count /
             len(df.index), neu_count / len(df.index)]
    colors = ['forestgreen', 'firebrick', 'goldenrod']
    plt.pie(sizes, labels=emotions, colors=colors,
            autopct='%1.1f%%', startangle=140)
    plt.title('Tweet frequency based on emotion')
    plt.axis('equal')
    plt.savefig('stats_images/emotion_pie.png')
    plt.close()


def iterate_words(counter, li, all_dict, emotion_dict):
    """ Iterate through words of the given list and add them to all_dict and emotion_dict, also increase the given counter. """

    counter += 1
    # iterate through the words in the list
    for word in li:
        # if word not in the dict of all words add it with frequency 1, else increase its frequency by 1
        if word not in all_dict:
            all_dict[word] = 1
        else:
            all_dict[word] += 1
        # if word not in the dict of words with certain emotion add it with frequency 1, else increase its frequency by 1
        if word not in emotion_dict:
            emotion_dict[word] = 1
        else:
            emotion_dict[word] += 1

    return counter


def visualize_stats(diction, plot_image_name, wordcloud_image_name, plot_title):
    """ Given a dictionary, visualize the statistics in horizontal-bar plot and wordcloud and save them as images. """

    # sort dictionary by values
    sorted_dict = OrderedDict(sorted(diction.items(), key=lambda t: t[1]))
    # get 20 first key-value pairs of sorted dict
    topdict = dict(list(sorted_dict.items())[-20:])

    # make horizontal-bar plots
    sns.set_style("darkgrid")
    ax = plt.figure().gca()
    ax.yaxis.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.barh(range(len(topdict)), list(
        topdict.values()), align='center')
    plt.yticks(range(len(topdict)), list(topdict.keys()))
    plt.xlabel('Frequency')
    plt.title(plot_title)
    # save figure to an image
    plt.savefig('stats_images/' + plot_image_name, bbox_inches="tight")
    plt.close()

    # make word clouds (maximum 100 words)
    wc = WordCloud(width=900, height=600, max_words=100, relative_scaling=1,
                   normalize_plurals=False, background_color='white').generate_from_frequencies(diction)
    plt.imshow(wc)
    plt.axis("off")
    # save cloud to an image
    wc.to_file('stats_images/' + wordcloud_image_name)
    plt.close()
