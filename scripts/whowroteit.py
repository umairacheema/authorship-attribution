#!/usr/bin/env python
# This script can be used to detect author's name from sample of his/her works
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import sys

#Model file path
model_file = '../data/classifier.pkl'
vectorizer_file = '../data/vectorizer.pkl'
# The machine learning model used in this case is only trained with the following authors.
famous_authors = ['charles_dickens','william_shakespeare','jane_austen','james_joyce','mark_twain','oscar_wilde','edgar_allan_poe',
                  'francis_bacon_st_albans','christopher_marlowe','joseph_conrad','agatha_christie','dh_lawrence']
#Function : print_help
#Purpose : Function to display help message
def print_help():
    print "Usage :"+sys.argv[0]+" <path to sample text file>"
    print " Where sample text file contains text by one of authors given above"
#Function : get_author_text
#Purpose : Convert raw text document to tokens
def get_author_text(sample_file):
     try:
        
        with open(sample_file,'r') as file:
            data = file.read()
     except IOError as e:
        print "I/O Error".format(e.errno, e.strerror)
        sys.exit(2)

     #Set language for stopwords
     stopwords_ = stopwords.words('english')
     #Instantiate Lemmatizer
     wordnet_lemmatizer_ = WordNetLemmatizer()
     #Clean the sample data
     contents = unicode(data, 'utf-8')
     prog = re.compile('[\t\n\r\f\v\d\']',re.UNICODE)
     contents = re.sub(prog,' ',contents).lower()
     #Remove punctuations
     prog=re.compile('[!\"#$%&\'()*+\,-./:;<=>?@[\]^_`{|}~]',re.UNICODE)
     contents = re.sub(prog,' ',contents)
     words = word_tokenize(contents)
     #Remove stop words and punctuations
     vocab = []
     for word in words:
         word=word.strip()
         if len(word)>1:
             if word not in stopwords_:
                 vocab.append(wordnet_lemmatizer_.lemmatize(word))
     return vocab

#Check input arguments
if (len(sys.argv) < 2):
    print_help()
    sys.exit(1)     
    
text = get_author_text(sys.argv[1])
clf = joblib.load(model_file)
svm = joblib.load('../data/svm.pkl')
vectorizer = joblib.load(vectorizer_file)
features = vectorizer.transform(text)
nb_prediction= clf.predict(features).tolist()
svm_prediction = svm.predict(features).tolist()
print nb_prediction
print svm_prediction

