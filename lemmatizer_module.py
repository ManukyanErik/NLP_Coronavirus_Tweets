import re
from nltk.stem import WordNetLemmatizer

class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
    def __call__(self, sentence):
        sentence=re.sub('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)',' ',sentence)
        sentence=re.sub('[^0-9a-z]',' ',sentence)
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word)>1]