import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import json
import math
import nltk
from nltk import word_tokenize
import re
import string
from gensim.models import FastText 
from gensim.models.word2vec import Word2Vec

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Conv1D, MaxPool1D, Flatten, Reshape, Concatenate, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

from sklearn.model_selection import train_test_split

nltk.download('reuters')
from nltk.corpus import reuters 

from multiprocessing import cpu_count


# List of documents
documents = reuters.fileids()


# List of categories
categories = reuters.categories();
num_categories = len(categories)


# Empty data structures
sentences = []
classes = []
len_classes = []


# Text pre-processing for each class
for cat_x in categories:
    category_docs = reuters.fileids(cat_x)
    n = len(category_docs)
    for id_x in range(1,n):
        document_words = reuters.words(category_docs[id_x])
        words = [word.lower() for word in document_words]
        words = [re.sub("[^a-zA-Z]+", "", word) for word in words]
        words = [word for word in words if word !='']
        len_classes.append(n)
        sentences.append(words)
        classes.append(categories.index(cat_x))


# Main dataframe
d = {'text':sentences,'category':classes,'len':len_classes}
df = pd.DataFrame(data = d)


# Dataframe filtering
volume_data = 200
df = df.sort_values('len', ascending=[False])
df = df[df['len']>volume_data]
df = df.sample(frac=1)


# Sampling function
def sampling_dataset(df,count):
    columns = df.columns
    class_df_sampled = pd.DataFrame(columns = columns)
    temp = []
    for c in df.category.unique():
        class_indexes = df[df.category == c].index
        random_indexes = np.random.choice(class_indexes, count, replace=False)
        temp.append(df.loc[random_indexes])
        
    for each_df in temp:
        class_df_sampled = pd.concat([class_df_sampled,each_df],axis=0)
    
    return class_df_sampled


# Sampling
df = sampling_dataset(df,volume_data)

# Getting dummies variables for Y
df_dummies = pd.get_dummies(df.category)
dummies = df_dummies.as_matrix()


# Model for word embeddings
num_features = 500 
model_deep = FastText(df.text, size=num_features, window=3, min_count=5, workers=cpu_count(),sg=1)
#model = Word2Vec(df.text, size=num_features, min_count=1, window=3, workers=4,sg=1)
#model.init_sims(replace=True)

df_size = len(df)
document_max_num_words = 50 #maximum size for text input
num_classes = len(df.category.unique())
print(num_classes)


# DataFrames
X = np.zeros(shape=(df_size, document_max_num_words, num_features))
Y = np.zeros(shape=(df_size, num_classes))
empty_word = np.zeros(num_features)


# Word and Doc embeddings
for idx, sentence in enumerate(df.text):
    Y[idx,:] = dummies[idx]
    for jdx, word in enumerate(sentence):
        if jdx == document_max_num_words:
            break

        else:
            if word in model_deep:
                X[idx, jdx, :] = model_deep[word]
            else:
                X[idx, jdx, :] = empty_word



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print(X_train.shape)


def show(y_true, y_pred):
    return y_true, y_pred


# Keras model architecture

filter_sizes = [3,4,5]
num_filters = 512
drop = 0.3

epochs = 10
batch_size = 100
num_channels = 1

inputs = Input(shape=(document_max_num_words,num_features))

conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]), padding='valid', kernel_initializer='normal', activation='relu')(inputs)
conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]), padding='valid', kernel_initializer='normal', activation='relu')(inputs)
conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]), padding='valid', kernel_initializer='normal', activation='relu')(inputs)

maxpool_0 = MaxPool1D(pool_size=(document_max_num_words - filter_sizes[0] + 1), strides=1, padding='valid')(conv_0)
maxpool_1 = MaxPool1D(pool_size=(document_max_num_words - filter_sizes[1] + 1), strides=1, padding='valid')(conv_1)
maxpool_2 = MaxPool1D(pool_size=(document_max_num_words - filter_sizes[2] + 1), strides=1, padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=num_classes, activation='softmax')(dropout) #initialization des poids de la matrice #gradual unfreezing



# Keras model creation


model = Model(inputs=inputs, outputs=output)
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))  # starts training

# Evaluate model
score, acc = model.evaluate(X_test, Y_test, batch_size=128)


print('Score: %1.4f' % score)
print('Accuracy: %1.4f' % acc)
