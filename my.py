import pandas as pd
import nltk
import numpy as np
from sklearn import *
import nltk
import re
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import (CountVectorizer,HashingVectorizer)
from sklearn.calibration import CalibratedClassifierCV
from time import strftime
from sklearn.externals import joblib


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




def main():

    file_id = strftime("%d%H%M%S")
    rootFolder = "/home/vdimir/edu/ml/avito"

    trainFileName = "%s/avito_train/avito_train.tsv" % rootFolder
    tetsFileName = "%s/avito_train/avito_test.tsv" % rootFolder
    output_file = "%s/avito_solution/avito_solution%s.csv" % (rootFolder,file_id)
    # pkl_out_file = "%s/avito_train_pkl/train_data.pkl" % rootFolder

    # --------

    dataTrain = pd.read_table(trainFileName, nrows=300000)
    dataTest = pd.read_table(tetsFileName)

    vectorizer = HashingVectorizer(analyzer = "word", tokenizer = None,\
            preprocessor = (lambda s: getWords(str(s))), binary = True)

    texts = (dataTrain['title']+" "+dataTrain['description']).apply(str)
    testTexts = (dataTest['title']+" "+dataTest['description']).apply(str)

    trainFeatures = vectorizer.fit_transform(texts)
    testFeatures = vectorizer.fit_transform(testTexts)

    swm = svm.LinearSVC(C=0.2)
    clf = CalibratedClassifierCV(swm)

    clf.fit(trainFeatures, dataTrain["is_blocked"])

    predicted_scores = clf.predict_proba(testFeatures).T[1]
    dataTest['res'] = predicted_scores
    dataTest_s = dataTest.sort_values('res', ascending=False)

    dataTest_s['itemid'].to_csv(output_file, index=False, header=['id'])


if __name__ == "__main__":
    main()


