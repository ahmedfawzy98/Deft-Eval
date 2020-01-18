import pandas as pd
from os import listdir
from os.path import join
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score,v

train_dir = 'training_set'
dev_dir = 'dev_set'

np.random.seed(500)

ss=3

def load_corpus():
    result = pd.DataFrame()
    for file in listdir(train_dir):
        output = pd.read_csv(join(train_dir, file), sep='\t', header=None)
        if 1 :
#            # Step - 1a : Remove blank rows if any.
            output[0].dropna(inplace=True)
            
    #            # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
            output[0] = [entry.lower() for entry in output[0]]
#                   TOKENIZE DATA
            output[0] = [word_tokenize(entry) for entry in output[0]]

            # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
            # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
            tag_map = defaultdict(lambda : wn.NOUN)
            tag_map['J'] = wn.ADJ
            tag_map['V'] = wn.VERB
            tag_map['R'] = wn.ADV
            
            for index,entry in enumerate(output[0]):
                # Declaring Empty List to store the words that follow the rules for this step
                Final_words = []
                # Initializing WordNetLemmatizer()
                word_Lemmatized = WordNetLemmatizer()
                # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
                for word, tag in pos_tag(entry):
                    # Below condition is to check for Stop words and consider only alphabets
                    if word not in stopwords.words('english') and word.isalpha():
                        word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                        Final_words.append(word_Final)
                # The final processed set of words for each iteration will be stored in 'text_final'
                
                output.loc[index,2] = str(Final_words)
            result = result.append(output, ignore_index = True)
#    print(result)
    
    return result

def load_test():
    result = pd.DataFrame()
    for file in listdir(dev_dir):
        output = pd.read_csv(join(train_dir, file), sep='\t', header=None)
        temp = ss-2
        if temp>0:
            temp-=1
#            # Step - 1a : Remove blank rows if any.
            output[0].dropna(inplace=True)
            
    #            # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
            output[0] = [entry.lower() for entry in output[0]]
#                   TOKENIZE DATA
            output[0] = [word_tokenize(entry) for entry in output[0]]

            # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
            # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
            tag_map = defaultdict(lambda : wn.NOUN)
            tag_map['J'] = wn.ADJ
            tag_map['V'] = wn.VERB
            tag_map['R'] = wn.ADV
            
            for index,entry in enumerate(output[0]):
                # Declaring Empty List to store the words that follow the rules for this step
                Final_words = []
                # Initializing WordNetLemmatizer()
                word_Lemmatized = WordNetLemmatizer()
                # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
                for word, tag in pos_tag(entry):
                    # Below condition is to check for Stop words and consider only alphabets
                    if word not in stopwords.words('english') and word.isalpha():
                        word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                        Final_words.append(word_Final)
                # The final processed set of words for each iteration will be stored in 'text_final'
                
                output.loc[index,2] = str(Final_words)
            result = result.append(output, ignore_index = True)
#    print(result)
    
    return result



  
            
Corpus = load_corpus()
test = load_test()    

Train_X = Corpus[2]
Train_Y = Corpus[1]
Test_X = test[2]
Test_Y = test[1]

# Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus[2])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",F1(predictions_SVM, Test_Y)*100)

print("Hello After Train")
#            docs += 1
#            items = doc[0].split()
#            check = doc[1]
#            if check:
#                deft_docs += 1
#            else:
#                no_deft_docs += 1
#            for item in items:
#                vocabulary.add(item)
#                if check:
#                    if item not in defenition_vocabs:
#                        defenition_vocabs[item] = 1
#                    else:
#                        defenition_vocabs[item] += 1
#                else:
#                    if item not in no_defenition_vocabs:
#                        no_defenition_vocabs[item] = 1
#                    else:
#                        no_defenition_vocabs[item] += 1