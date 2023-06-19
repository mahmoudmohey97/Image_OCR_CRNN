from keras.utils import pad_sequences
corpus = "اآأدذجحخهعغفقثصضطكمنتلبيسشظزوةىلارؤءئ "

# Function to transform text to numbers using given string corpus
def encode_to_labels(text, corpus=corpus):
    char2idx = [corpus.index(char) for char in text]
    return char2idx

# Function to transform numbers to text using given string corpus
def labels2text(label_array, corpus=corpus):
    text = ''.join([corpus[token_id] for token_id in label_array if token_id != -1])
    return text

# Function to pad text with value of corpus length
def pad_text(text, max_length, pad_value):
    return pad_sequences(text, maxlen=max_length, value=pad_value, padding='post')

