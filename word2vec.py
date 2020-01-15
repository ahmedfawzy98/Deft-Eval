import pandas as pd
from os import listdir
from os.path import join
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

train_dir = 'training_set'
dev_dir = 'dev_set'


def load_corpus():
    corpus = ''
    for file in listdir(train_dir):
        output = pd.read_csv(join(train_dir, file), sep='\t', header=None).to_numpy()
        for doc in output:
            corpus += doc[0]

    corpus = corpus.lower()
    corpus = re.sub('[^a-zA-Z]', ' ', corpus)
    corpus = re.sub(r'\s+', ' ', corpus)

    all_sentences = nltk.sent_tokenize(corpus)
    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
    for i in range(len(all_words)):
        all_words[i] = [word for word in all_words[i] if word not in stopwords.words('english')]
    return all_words

def load_test_data():
    result = np.array([])
    for file in listdir(dev_dir):
        output = pd.read_csv(join(dev_dir, file), sep='\t', header=None).to_numpy()
        break
    result = np.concatenate((result, output), axis=None)
    return result

if __name__ == "__main__":
    all_words = load_corpus()
    word2vec = Word2Vec(all_words, min_count=1)
    # vocabulary = word2vec.wv.vocab
    # print(vocabulary)
    # print(word2vec.wv['physics'])
    # similar_words = word2vec.wv.most_similar('physics')
    # print(similar_words)
    # dissimilar_words = word2vec.doesnt_match('physics is more science'.split())
    # print(dissimilar_words)

    result = load_test_data()
    for i in range(0, len(result), 2):
        print(result[i], end='\n')
        print('Dissimilar words in this sentence: ', end='')
        print(word2vec.wv.doesnt_match(result[i].split()), end='\n')
