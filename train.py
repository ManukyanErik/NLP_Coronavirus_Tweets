import re
import pandas as pd
import pickle

from nltk.corpus import stopwords

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Embedding, GlobalMaxPool1D

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from lemmatizer_module import Lemmatizer

train = pd.read_csv('./data/Corona_NLP_train.csv', encoding='latin-1')

def change_sen(sentiment):
    # be careful and run this function only once
    if sentiment == "Extremely Positive":
        return 'positive'
    elif sentiment == "Extremely Negative":
        return 'negative'
    elif sentiment == "Positive":
        return 'positive'
    elif sentiment == "Negative":
        return 'negative'
    else:
        return 'netural'
    
train['Sentiment']=train['Sentiment'].apply(lambda x:change_sen(x))

# load stop words
stop_word = stopwords.words('english')

# Cleaning the tweet
def clean(text):
    #     remove urls
    text = re.sub(r'http\S+', " ", text)
    #     remove mentions
    text = re.sub(r'@\w+',' ',text)
    #     remove hastags
    text = re.sub(r'#\w+', ' ', text)
    #     remove digits
    text = re.sub(r'\d+', ' ', text)
    #     remove html tags
    text = re.sub('r<.*?>',' ', text) 
    #     remove stop words 
    text = text.split()
    text = " ".join([word for word in text if not word in stop_word])
        
    return text


train['OriginalTweet'] = train['OriginalTweet'].apply(lambda x: clean(x))
train = train.iloc[:,4:]

# Encode labels in column 'Sentiment'.
label_encoder = preprocessing.LabelEncoder()
train['Sentiment']= label_encoder.fit_transform(train['Sentiment'])
  

train_text, val_text, train_label, val_label = train_test_split(train.OriginalTweet, train.Sentiment, test_size=0.3, random_state=42)

# Tokenization 
token=Tokenizer(num_words=5000, oov_token=Lemmatizer())
token.fit_on_texts(train_text)

# Save the tokenizer to a file using pickle
with open('./tokenizers/tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(token, tokenizer_file)

train_sequences_padded=token.texts_to_sequences(train_text)
train_sequences_padded=pad_sequences(train_sequences_padded, maxlen=60, padding='post', truncating='post')

val_sequences_padded=token.texts_to_sequences(val_text)
val_sequences_padded=pad_sequences(val_sequences_padded, maxlen=60, padding='post', truncating='post')

# LSTM
# The model will stop training if the validation accuracy does not improve for 3 consecutive epochs.
early_stop = EarlyStopping(monitor='val_accuracy', patience=3)                                      
# If the validation accuracy does not improve for 2 consecutive epochs, the learning rate will be reduced
reduceLR = ReduceLROnPlateau(monitor='val_accuracy', patience=2)

# Model architecture
embedding_dimension=32
vocab_size=len(token.word_index)
model=Sequential()
model.add(Input(shape=(60,)))
model.add(Embedding(vocab_size+1, embedding_dimension))
model.add(LSTM(64,return_sequences=True))
model.add(GlobalMaxPool1D())
model.add(Dense(64))
model.add(Dense(3,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

lstm = model.fit(train_sequences_padded, train_label, validation_data=(val_sequences_padded, val_label),
                 epochs=50,
                 batch_size=64,
                 callbacks=[reduceLR, early_stop])

model.save('./models/lstm.h5')
