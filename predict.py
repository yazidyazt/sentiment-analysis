from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
import pickle
import numpy as np

# Import model
model = load_model('models/model.h5')
# Import Tokenizer
with open('models/tokenizer.pkl', 'rb') as tok:
    tokenizer = pickle.load(tok)

EMBEDDING_DIM = 100
MAXLEN = 16
TRUNCATING = 'post'
PADDING = 'post'
OOV_TOKEN = "<OOV>"

def seq_pad_and_trunc(sentences, tokenizer, padding, truncating, maxlen):
    sequences = tokenizer.texts_to_sequences(sentences)
    pad_trunc_sequences = pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating)
    return pad_trunc_sequences

def predict_comment(tokenizer, comment, cat):
    comment_seq = tokenizer.texts_to_sequences([comment])
    comment_pad = pad_sequences(comment_seq, maxlen=MAXLEN, padding=PADDING)
    pred_prob = model.predict(comment_pad)
    pred_labels = np.argmax(pred_prob)
    predicted_category = cat[pred_labels]
    return pred_prob, predicted_category, pred_labels
