#%config IPCompleter.greedy=True


# -*- coding: utf-8 -*-
# Name : language_detect
#This jupyter notebook can be used to detect one of the 20 languages - bulgarian, czech, danish, german, greek, spanish
#     estonian, finnish, french, hungarian, italian, lithuanian, latvian, dutch, polish, portugese, romanian, slovakian,
#     swedish and slovenian
# The corpus is taken from https://www.statmt.org/europarl/
# the corpus is taken, unzipped and renamed with number file extensions like .1 .2 .3 to encode languages in float.
# the language code follows https://en.wikipedia.org/wiki/ISO_639 
#system libraries
import os
from pathlib import Path

#data manipulation libraries
import pandas as pd
import numpy as np

#machine learning library - scikit-learn
from sklearn.neighbors import KNeighborsClassifier
#place all the train&test data in a folder "collected" with the extension as the language code 
#For convenience i have uploaded the traintest data into my google drive and made public , 
# https://drive.google.com/open?id=1JkA84vAwP4kR1P48sIvRboBKGjWT9wj6
#just download and unzip and place it same location as this jupyter / python file  
dataFolder='./collected'
constNum_of_lines_selected = 5000
floatSplitRatio=0.8
datafolder = Path(dataFolder)


def loadfile(file):
    with open(file, 'r', encoding="utf-8") as fileObj:
        return(fileObj.read())

def selectQualityText(dFrame):
    #Select only high text lines from the file 
    dFrame = dFrame.loc[dFrameIntermdiate['length'] > 200]
    #filter to top n lines to make training easier and faster
    #this parameter can be increased if you are patient enough and can afford large computer power 
    dFrame = dFrame.head(constNum_of_lines_selected)
    return dFrame

def constructDataFrame(lineSeparatedText):
    #create the dataframe with a text column, lang identifier column and length column
    dFrame = pd.DataFrame()
    dFrame['text'] = pd.Series(lineSeparatedText.split('\n'))
    dFrame['text'] = [i.strip() for i in dFrame['text']]
    dFrame['ltext'] = [i.lower() for i in dFrame['text']]
    dFrame['iso639code'] = (file.name.title().split('.')[-1])
    dFrame = dFrame.sample(frac=1).reset_index(drop=True)
    dFrame['length'] = [len(i) for i in dFrame['text']]
    return dFrame
    
dfText = pd.DataFrame(columns=['text', 'iso639code'])

for file in datafolder.iterdir():
    print("processing file",file)
    #if not file then ignore the directory
    if not os.path.isfile(file):
        continue
    text_inside = '\n'.join([loadfile(file)])
    dFrameIntermdiate = constructDataFrame(text_inside)
    dFrameIntermdiate = selectQualityText(dFrameIntermdiate)
    dfText = dfText.append(dFrameIntermdiate.copy(), ignore_index=True)
    print("numer of text lines created for training/testing so far=",dfText.shape[0])

def split_train_test(dFrame, ratio=0.5):
    """default split of 50/50 but can be set when calling using constant floatSplitRatio"""
    #First shuffle the dataframe before splitting
    dFrame = dFrame.sample(frac=1).reset_index(drop=True)
    traindFrame = dFrame[:][:int(dFrame.shape[0] * ratio)]
    testdFrame = dFrame[:][int(dFrame.shape[0] * ratio):]
    return traindFrame, testdFrame

def calRelRatio(dFrame):
    """Calculate the relative presence of each alphabet in all 20 languages in each text
       as a ratio.
    """
    allAlphabets = list(''.join({character for character in ''.join(dFrame['ltext']) if character.isalpha()}))
    for alphabet in allAlphabets:
        dFrame[alphabet] = [r.count(alphabet) for r in dFrame['ltext']] / dFrame['length']
    return allAlphabets, dFrame

# could be speeden up
globalCharList, dfText = calRelRatio(dfText)
computedDFrame = pd.DataFrame()
computedDFrame['iso639code'] = dfText['iso639code']
for character in globalCharList:
    computedDFrame[character] = dfText[character]
traindFrame, testdFrame = split_train_test(computedDFrame, ratio=floatSplitRatio)
print(traindFrame)

#the knn classifier expects a numpy array of features and labels for train and tests
x_train = np.array([np.array(row[1:]) for index, row in traindFrame.iterrows()])
y_train = np.array([label for label in traindFrame['iso639code']])
x_test = np.array([np.array(row[1:]) for index, row in testdFrame.iterrows()])
y_test = np.array([label for label in testdFrame['iso639code']])

def train(x_train, y_train, k):
    """returns a knn classifier"""
    clf = KNeighborsClassifier(k)
    clf.fit(x_train, y_train)
    return clf
 
def test(clf, x_test, y_test):
    '''tests the classifier'''
    predictions = clf.predict(x_test)
    accuracy = len([i for i in range(len(y_test)) if y_test[i] == predictions[i]]) / len(y_test)
    return accuracy

#I use the euclidean metric which is 2. one could use multiple values to find an optimal one
clf = train(x_train, y_train, 2)
accuracy = test(clf, x_test, y_test)
print("accuracy is "+ str(round(accuracy * 100, 2)) + '%')

langdict = {"1":"German","2":"Danish","3":"Czech","4":"Bulgarian","5":"Greek","6":"Spanish","7":"Estonian", "8":"Finnish",
           "9":"French","10":"Hungarian","11":"Italian","12":"Lithuanian","13":"Latvian","14":"Dutch","15":"Polish","16":"Portugese"
           ,"17":"Romanian", "18":"Slovak", "19": "Slovenian", "20":"Swedish"}
def featurizeTestData(text):
    ratios = []
    for alphabet in globalCharList:
        ratios.append(text.count(alphabet) / len(text))
    return np.array(ratios)

def predictTestData(text, clf=clf):
    text_features = featurizeTestData(text)
    langcode  = clf.predict(np.array(np.array([text_features])))[0]
    return langdict[langcode]

#This is the place where u can test my model
#to test any doc, enter one of the 20 language texts in a document named testdoc.txt 
#and place in the same folder as this jupyter notebook
file = "./testdoc.txt"
test_text = ""
with open(file, 'r', encoding="utf-8") as f:
    for i in range(1,11):
        test_text = test_text  + f.readline()
print(predictTestData(test_text.lower()))



#If you feel to test directly use below snippet and comment just above line of code
#test_text = "Die FDP-Spitze hat Parteichef Christian Lindner nach seinem Kriseneinsatz in Thüringen mit deutlicher Mehrheit das Vertrauen ausgesprochen. Linder erhielt von 36 abgegebenen Stimmen 33 Ja-Stimmen und"
#predict_lang(test_text)


