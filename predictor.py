from keras.models import model_from_json
from keras.models import load_model
from gensim.models import FastText
from nltk import word_tokenize
from keras.models import Model
import pandas as pd
import numpy as np
import h5py as h5
import string
import json
import re
import os



files_names = ['administratif', 'nettoy_modif', 'nettoy_qualite', 'nettoy_absence', 'autres', 'démarrage', 'organisation_de_la_prestation']
sizes = [153,160,151,79,212,82,190]*2


def text_cleaning(text):

    # input : string (textual data)
    # output : list of cleaned tokens

    tokens = word_tokenize(text.lower()) #lower case only
    words = [re.sub("[^a-zA-Z]+", "", word) for word in tokens] #keeping only letters but delete accents, not good for french
    words = [word for word in words if word !=''] #deleting empty
    words = [word for word in words if len(word) < 20] #deleting bizarre words
    return words



def text_embedding(text, df_size, document_max_num_words, num_features, model_deep):

    words = text_cleaning(text)

    X = np.zeros(shape=(document_max_num_words, num_features))
    empty_word = np.zeros(num_features)

    for idx, word in enumerate(words):
        if idx == document_max_num_words:
            break
        else:
            if word in model_deep:
                X[idx, :] = model_deep[word]
            else:
                X[idx, :] = empty_word

    return X



def models_output(text, sizes, document_max_num_words, num_features, model_deep, files_names):

    outputs = []

    compt = 0

    for index, name in enumerate(files_names):

        X = text_embedding(text, sizes[index], document_max_num_words, num_features, model_deep)

        json_string = json.dumps(json.load(open("binary_classifier"+"_"+name+".json"))) #load json and then convert it to string
        model = model_from_json(json_string)
        
        model.load_weights("weights"+"_"+name+".h5")
        pred = model.predict(np.array( [X,] ))

        print(compt)

        compt = compt + 1

        outputs.append(pred.item(0))

    max1 = max(outputs)
    index1 = outputs.index(max1)

    max2 = sorted(outputs)[-2]
    index2 = outputs.index(max2)

    max3 = sorted(outputs)[-3]
    index3 = outputs.index(max3)

    max4 = sorted(outputs)[-4]
    index4 = outputs.index(max4)

    predicted1_tag = files_names[index1]
    predicted2_tag = files_names[index2]
    predicted3_tag = files_names[index3]
    predicted4_tag = files_names[index4]


    return predicted1_tag, predicted2_tag



text = "Bonsoir, Personne n'est passé dans mes locaux aujourd'hui et le ménage n'a pas été fait... Merci de me contacter"


model_deep = FastText.load("./fasttext.model")
outputs = models_output(text,sizes,24430,200,model_deep,files_names)
print(outputs)

    



