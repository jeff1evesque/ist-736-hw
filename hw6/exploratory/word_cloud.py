#!/usr/bin/python

from nltk.corpus import stopwords as stp
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def word_cloud(
    df,
    filename='plot.png',
    width=2000,
    height=1250,
    background_color='black',
    stopwords=[],
    show=False
):
    '''

    generate word cloud using provided dataframe.

    '''

    # extend stopwords
    stopwords_english = set(stp.words('english'))
    stopwords.extend(stopwords_english)

    # generate wordcloud
    text = df.values
    wordcloud = WordCloud(
        width = width,
        height = height,
        background_color = background_color,
        stopwords = stopwords
    ).generate(str(text))

    fig = plt.figure(
        figsize = (40, 30),
        facecolor = 'k',
        edgecolor = 'k'
    )

    # generate plot
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)

    # save plot
    plt.savefig(filename)

    if show:
        plt.show()
    else:
        plt.close()
