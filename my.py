# coding: utf-8

import pandas as pd
import nltk
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm

import nltk
import re
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import (CountVectorizer,HashingVectorizer)
from sklearn.calibration import CalibratedClassifierCV
from time import strftime
from sklearn.externals import joblib


import sys

import logging
logging.basicConfig(format=u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level=logging.NOTSET)


stopwords = frozenset(word for word in nltk.corpus.stopwords.words("russian") if word != u"не")
stemmer = SnowballStemmer('russian')
engChars = [ord(char) for char in u"cCyoOBaAKpPeE"]
rusChars = [ord(char) for char in u"сСуоОВаАКрРеЕ"]
eng_rusTranslateTable = dict(zip(engChars, rusChars))
rus_engTranslateTable = dict(zip(rusChars, engChars))

def correctWord(w):
    """ Corrects word by replacing characters with written similarly depending on which language the word.
        Fraudsters use this technique to avoid detection by anti-fraud algorithms."""

    if len(re.findall(u"[а-я]", w)) > len(re.findall(u"[a-z]", w)):
        return w.translate(eng_rusTranslateTable)
    else:
        return w.translate(rus_engTranslateTable)

def getWords(text, stemmRequired=False, correctWordRequired=False):
    cleanText = re.sub(u'[^a-zа-я0-9]', ' ', text.lower())
    if correctWordRequired:
        words = [correctWord(w) if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(correctWord(w)) for w
                 in cleanText.split() if len(w) > 1 and w not in stopwords]
    else:
        words = [w if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(w) for w in cleanText.split() if
                 len(w) > 1 and w not in stopwords]

    return " ".join(words)

class ItemProcessor:
    def __init__(self):
        self.itemsProcessed = 0
        self.totalItems = 0

    def reset(self, total=0):
        self.itemsProcessed = 0
        self.totalItems = total

    def getWords(self,s, stemmRequired=False, correctWordRequired=False):
        self.itemsProcessed += 1
        if (self.itemsProcessed % 1000 == 0):
            p = self.itemsProcessed/ self.totalItems if self.totalItems != 0 else -1
            logging.info("%d/%d items processed (%f%%)" % (self.itemsProcessed, self.totalItems, p*100))
        return getWords(str(s), stemmRequired, correctWordRequired)


root_folder = "."


def genFratures(pkl_out_file, stemm, correctWord, numrows):
    train_file_name = "%s/avito_train/avito_train.tsv" % root_folder
    tets_file_name = "%s/avito_train/avito_test.tsv" % root_folder

    logging.info("ReadData...")

    dataTrain = pd.read_table(train_file_name, nrows=numrows)
    dataTest = pd.read_table(tets_file_name)
    testItemIds = dataTest['itemid']

    trainTargets = dataTrain["is_blocked"]
    p = ItemProcessor()
    vectorizer = HashingVectorizer(analyzer = "word", tokenizer = None,\
            preprocessor = (lambda s: p.getWords(s,stemm, correctWord)), binary = True)


    texts = (dataTrain['title']+" "+dataTrain['description']).apply(str)
    testTexts = (dataTest['title']+" "+dataTest['description']).apply(str)


    logging.info("vectorizing dataTest...")
    p.reset(len(texts))
    trainFeatures = vectorizer.fit_transform(texts)

    logging.info("vectorizing dataTest...")
    p.reset(len(testTexts))
    testFeatures = vectorizer.fit_transform(testTexts)

    logging.info("dump data to %s" % pkl_out_file)
    joblib.dump((trainFeatures, trainTargets, testFeatures, testItemIds), pkl_out_file)

    return (trainFeatures, trainTargets, testFeatures, testItemIds)

def processCmArgs():
    numrows=300000
    stemm=False
    correctWord=False
    generateFratures = False
    for (i,a) in enumerate(sys.argv):
        if (a == "-s"):
            stemm=True
        if (a == "-c"):
            correctWord = True
        if (a == "-g"):
            generateFratures = True
        if (a == "-n"):
            numrows=int(sys.argv[i+1])
    return (stemm, correctWord, numrows, generateFratures)

def main():
    (stemm, correctWord, numrows, generateFratures) = processCmArgs()
    logging.info("params: stemm %s, correct=%s numrows=%s gen=%s" % (stemm, correctWord, numrows, generateFratures))

    file_id = strftime("%d%H%M%S")
    output_file = "%s/avito_solution/avito_solution%s.csv" % (root_folder,file_id)

    pkl_out_file = "%s/avito_train_pkl/num-%s_stemm-%s_corr-%s.pkl" % (root_folder,numrows,stemm,correctWord)

    if generateFratures:
        (trainFeatures, trainTargets, testFeatures, testItemIds) = genFratures(pkl_out_file,stemm, correctWord, numrows)
    else:
        logging.info("read dump from %s" % pkl_out_file)
        (trainFeatures, trainTargets, testFeatures, testItemIds) = joblib.load(pkl_out_file)

    logging.info("Fitting...")

    swm = svm.LinearSVC(C=0.3)
    clf = CalibratedClassifierCV(swm)

    clf.fit(trainFeatures, trainTargets)

    logging.info("Predicting...")

    predicted_scores = clf.predict_proba(testFeatures).T[1]

    dataPred = pd.DataFrame()

    dataPred['res'] = predicted_scores
    dataPred['itemid'] = testItemIds
    logging.info("Sotring...")

    dataPred = dataPred.sort_values('res', ascending=False)

    logging.info("write result to %s..." % output_file)

    dataPred['itemid'].to_csv(output_file, index=False, header=['id'])


if __name__ == "__main__":
    main()


