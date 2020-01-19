import pandas as pd
from os import listdir
from os.path import join
import numpy as np
import re
import nltk
import copy
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_dir = 'training_set'
dev_dir = 'dev_set'


def load_corpus():
    labels = []
    corpus = ''
    # all_words = []
    for file in listdir(train_dir):
        document = pd.read_csv(join(train_dir, file), sep='\t', header=None).to_numpy()
    #     for line in document:
    #         sentences = nltk.sent_tokenize(line[0])
    #         for sentence in sentences:
    #             sentence = sentence.lower()
    #             sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    #             sentence = re.sub(r'\s+', ' ', sentence)
    #         words = [nltk.word_tokenize(sentence) for sentence in sentences]
    #         for i in range(len(words)):
    #             words[i] = [word for word in words[i] if word not in stopwords.words('english')]
    #             for word in words[i]:
    #                 labels.append(str(line[1]))
    # write_labels(labels)
    # return all_words
    
    # return corpus
        for line in document:
            corpus += line[0]

    corpus = corpus.lower()
    corpus = re.sub('[^a-zA-Z]', ' ', corpus)
    corpus = re.sub(r'\s+', ' ', corpus)

    all_sentences = nltk.sent_tokenize(corpus)
    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
    for i in range(len(all_words)):
        all_words[i] = [word for word in all_words[i] if word not in stopwords.words('english')]
    write_all_words(all_words)

def write_labels(labels):
    file = open('labels.txt', 'w')
    for label in labels:
        file.write(label + '\n')
    file.close()

def load_labels():
    labels = []
    file = open('labels.txt', 'r')
    for line in file.readlines():
        labels.append(line[:-1])
    file.close()
    return labels

def write_all_words(all_words):
    file = open('all_words.txt', 'w')
    for words in all_words:
        for word in words:
            file.write(word + '\n')
    file.close()

def load_all_words():
    words = []
    file = open('all_words.txt', 'r')
    for line in file.readlines():
        words.append(line[:-1])
    file.close()
    return words

def load_test_data():
    corpus = ''
    for file in listdir(dev_dir):
        document = pd.read_csv(join(dev_dir, file), sep='\t', header=None).to_numpy()
        for line in document:
                corpus += line[0]

    corpus = corpus.lower()
    corpus = re.sub('[^a-zA-Z]', ' ', corpus)
    corpus = re.sub(r'\s+', ' ', corpus)

    all_sentences = nltk.sent_tokenize(corpus)
    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
    for i in range(len(all_words)):
        all_words[i] = [word for word in all_words[i] if word not in stopwords.words('english')]
    return all_words


if __name__ == "__main__":
    load_corpus()
    all_words = load_all_words()
    labels = load_labels()
    # print(labels)
    # print(all_words)
    # test_data = load_test_data()
    # model = Word2Vec(all_words, min_count=1)
    model = Word2Vec.load('word2vec.model')
    train_vectors = [model[word] for word in all_words]
    # test_vectors = [model[word] for words in all_words for word in words]
    xtrain, xtest, ytrain, ytest = train_test_split(train_vectors, labels, test_size=0.25, random_state=0)
    sc_x = StandardScaler()
    xtrain = sc_x.fit_transform(xtrain)
    xtest = sc_x.transform(xtest)
    classifier = LogisticRegression(random_state=0)
    classifier.fit(xtrain, ytrain)
    prediction = classifier.predict(xtest)
    print(prediction)
    # cm = confusion_matrix(y_test, prediction)
    # print("Confusion Matrix : {}".format(cm))