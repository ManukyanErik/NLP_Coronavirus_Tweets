import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.utils import pad_sequences
from keras.models import load_model
from lemmatizer_module import Lemmatizer
from sklearn.metrics import classification_report

test = pd.read_csv('./data/Corona_NLP_test.csv', encoding='latin-1')

# Load the tokenizer from the saved file
with open('./tokenizers/tokenizer.pkl', 'rb') as tokenizer_file:
    loaded_token = pickle.load(tokenizer_file)

test_sequences_padded=loaded_token.texts_to_sequences(test['OriginalTweet'])
test_sequences_padded=pad_sequences(test_sequences_padded, maxlen=60, padding='post', truncating='post')

# Load the saved model
loaded_model = load_model('./models/lstm.h5')

# Assuming 'model' is your Sequential model
predictions = loaded_model.predict(test_sequences_padded)

# Get the predicted classes by finding the index of the maximum value for each prediction
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)
