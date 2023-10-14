# NLP Classification Project: Coronavirus Tweets

This project involves classifying tweets related to the coronavirus using Natural Language Processing (NLP) techniques. The objective is to categorize tweets into sentiment classes: positive, negative, or neutral.

## Project Structure

The project is organized into the following directories and files:

- `data`: Contains the dataset files for training and testing.
- `models`: Stores the trained model.
- `tokenizers`: Holds the tokenizer used for text processing.
- `train.py`: Python script for training the NLP classification model.
- `predict.py`: Python script for making predictions using the trained model.
- `lemmatizer_module.py`: Python script containing a lemmatizer module used for text preprocessing.

## Usage

### 1. Training the Model (`train.py`)

To train the NLP classification model, execute the `train.py` script. This script performs the following steps:

1. **Data Preprocessing:**
   - Read the training dataset (`Corona_NLP_train.csv`) and preprocesses the tweet data.
   - Clean the tweets by removing URLs, mentions, hashtags, digits, HTML tags, and stop words.
   - Convert sentiment labels to numerical values using label encoding.

2. **Tokenization and Padding:**
   - Tokenize the preprocessed text using the Keras Tokenizer.
   - Pad the tokenized sequences to ensure consistent lengths for training.

3. **Model Architecture:**
   - Define the LSTM model architecture using Keras, including embedding layers, LSTM layers, and dense layers.
   - Compile the model using Adam optimizer and sparse categorical cross-entropy loss function.

4. **Training:**
   - Train the model using the training data and validates using a validation set.
   - Use early stopping and learning rate reduction callbacks for optimization.

5. **Saving:**
   - Save the trained model to the `models` directory in an HDF5 file format.
   - Save the tokenizer to the `tokenizers` directory using pickle for later use during prediction.

Run the following command to train the model:

```bash
python train.py
```

### 2. Making Predictions (`predict.py`)

To make predictions on new data using the trained model, use the `predict.py` script. This script performs the following steps:

1. **Loading Pretrained Model and Tokenizer:**
   - Load the pre-trained LSTM model and tokenizer from the `models` and `tokenizers` directories, respectively.

2. **Data Preprocessing:**
   - Preprocess the testing dataset (`Corona_NLP_test.csv`) similarly to the training data, including text cleaning and tokenization.

3. **Prediction:**
   - Use the loaded model to predict the sentiment classes for the preprocessed testing data.
   - Print the predicted sentiment classes to the console.

Run the following command to make predictions:

```bash
python predict.py
```

## Additional Information

- The `lemmatizer_module.py` file contains a custom lemmatizer class using NLTK's WordNet lemmatizer. It performs lemmatization on text data, which is a process of reducing words to their base or root form. This helps in text preprocessing to improve the quality of the input data for the NLP model.
- The `train.py` script uses the provided training dataset `Corona_NLP_train.csv` located in the `data` directory for training the model.
- The `predict.py` script uses the provided testing dataset `Corona_NLP_test.csv` located in the `data` directory to evaluate the trained model and print the predicted sentiment classes.

---
