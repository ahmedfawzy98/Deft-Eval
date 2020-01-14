import pandas as pd
from os import listdir
from os.path import join
import numpy as np

train_dir = 'training_set'
dev_dir = 'dev_set'

defenition = 1
no_defenition = 0

vocabulary = set()
defenition_vocabs = {}
no_defenition_vocabs = {}
docs = 0
deft_docs = 0
no_deft_docs = 0

def load_corpus():
    global docs, deft_docs, no_deft_docs
    for file in listdir(train_dir):
        output = pd.read_csv(join(train_dir, file), sep='\t', header=None).to_numpy()
        for doc in output:
            docs += 1
            items = doc[0].split()
            check = doc[1]
            if check:
                deft_docs += 1
            else:
                no_deft_docs += 1
            for item in items:
                vocabulary.add(item)
                if check:
                    if item not in defenition_vocabs:
                        defenition_vocabs[item] = 1
                    else:
                        defenition_vocabs[item] += 1
                else:
                    if item not in no_defenition_vocabs:
                        no_defenition_vocabs[item] = 1
                    else:
                        no_defenition_vocabs[item] += 1
            
def load_test_data():
    result = np.array([])
    for file in listdir(dev_dir):
        output = pd.read_csv(join(dev_dir, file), sep='\t', header=None).to_numpy()
        result = np.concatenate((result, output), axis=None)
    return result

def contain_defenition(words):
    deft_value = deft_docs / docs
    no_deft_value = no_deft_docs / docs
    for word in words:
        deft_value *= (defenition_vocabs.get(word, 0) + 1) / (deft_docs + len(vocabulary))
        no_deft_value *= (no_defenition_vocabs.get(word, 0) + 1) / (no_deft_docs + len(vocabulary))
    return deft_value > no_deft_value
        

def naive_byes_classify(test_data):
    right_classify = 0
    sentences = 0
    for i in range(0, np.size(test_data), 2):
        sentences += 1
        if contain_defenition(test_data[i].split()):
            print("This sentence has defenition.   {}".format(test_data[i + 1]))
            if test_data[i + 1] == defenition:
                # print('hit')
                right_classify += 1
        else:
            print('This sentence doesn\'t have defenition.   {}'.format(test_data[i + 1]))
            if test_data[i + 1] == no_defenition:
                # print('hit')
                right_classify += 1
    print('Accuracy = {:.2f}%'.format(right_classify / sentences * 100))

if __name__ == "__main__":
    load_corpus()
    test_data = load_test_data()
    naive_byes_classify(test_data)
    