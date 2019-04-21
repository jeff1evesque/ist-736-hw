#!/usr/bin/python

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def word_cloud(
    df,
    filename='plot.png',
    width=2000,
    height=1250,
    background_color='black'
):
    '''

    generate word cloud using provided dataframe.

    '''

    text = df.values
    wordcloud = WordCloud(
        width = width,
        height = height,
        background_color = background_color,
        stopwords = STOPWORDS
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
    plt.show()
