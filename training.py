import numpy as np 
import json
import tensorflow as tf
from tensorflow import _keras , keras , _keras_package , _kernel_dir , _keras_module
import colorama
import nltk
import pickle
import random
from keras.models import Sequential
from keras.layers import Dense,Embedding,GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """
    Pad sequences to the same length.

    Args:
        sequences (list of list of int): List of input sequences.
        maxlen (int, optional): Maximum length of sequences. If not provided, the maximum length of sequences is used. Defaults to None.
        dtype (str, optional): Data type of the output sequences. Defaults to 'int32'.
        padding (str, optional): 'pre' or 'post', specifies whether to pad sequences before or after each sequence. Defaults to 'pre'.
        truncating (str, optional): 'pre' or 'post', specifies whether to truncate sequences before or after each sequence. Defaults to 'pre'.
        value (float, optional): Float or string, padding value. Defaults to 0.

    Returns:
        numpy.ndarray: Padded sequences.

    Raises:
        ValueError: If `padding` or `truncating` is not one of 'pre' or 'post'.
    """
    # Find the length of the longest sequence if maxlen is not provided
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)

    # Initialize an empty array to store the padded sequences
    padded_sequences = np.full((len(sequences), maxlen), value, dtype=dtype)

    # Pad or truncate sequences based on the specified parameters
    for i, seq in enumerate(sequences):
        if truncating == 'pre':
            truncated_seq = seq[-maxlen:]
        elif truncating == 'post':
            truncated_seq = seq[:maxlen]
        else:
            raise ValueError("Truncating type must be one of 'pre' or 'post'")
        
        if padding == 'pre':
            padded_sequences[i, -len(truncated_seq):] = truncated_seq
        elif padding == 'post':
            padded_sequences[i, :len(truncated_seq)] = truncated_seq
        else:
            raise ValueError("Padding type must be one of 'pre' or 'post'")

    return padded_sequences

training_sentences = []
training_labels = []
labels=[]
responses= []


with open('intents.json') as file: # reading the json file
    data= json.load(file)
    
for intent in data['intents']: #it reads the data from the json file
    for pattern in intent['patterns']:# it extracts the patterns fromm the json file
        training_sentences.append(pattern) # appending the patterns to the training_sentences list
        training_labels.append(intent['tag'])#appending the tags to the training_labels list
    responses.append(intent['responses'])#appending the responses to the responses list
    
    if intent['tag'] not in labels :
        labels.append(intent['tag'])#appending the tags seperately to the labels list

num_classes = len(labels)


lblencoder = LabelEncoder() # calling the label encoder function from scikit-learn
lblencoder.fit(training_labels) #converts training_labels to numerical values
training_labels = lblencoder.transform(training_labels) #used to transform the labels to y

vocab_size = 1000
embedding_dim =16
OOV_token = '<OOV>'
max_len = 20

tokenizer = Tokenizer(num_words=vocab_size, oov_token=OOV_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Define the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, np.array(training_labels), epochs=1000)

# Save the model
model.save('chatbot_model.h5')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lblencoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
    

