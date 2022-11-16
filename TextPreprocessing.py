#################################################################################################
#                                                                                               #
# Preprocessing functions that can be useful for NLP projects.  I have broken out all of the    #
# individual functions so each can be independently tested if we want to.  I have created a     #
# normalize function that does the basic combination of proprocessing steps like lowercase,     #
# whitespace cleaning, etc.  I then created a Preprocessing function that uses normalize and    #
# then removes the stopwords and lemmatizes.                                                    #
#################################################################################################

# import the necessary libraries
import re
import string

import contractions
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')


# Lowercase the text function
def text_lowercase(text):
    return text.lower()


# Remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result


# convert numbers into words
def convert_number(text):
    inflect_engine = inflect.engine()

    # split string into list of words
    temp_str = text.split()
    # initialise empty list
    new_string = []

    for word in temp_str:
        # if word is a digit, convert the digit
        # to numbers and append into the new_string list
        if word.isdigit():
            temp = inflect_engine.number_to_words(word)
            new_string.append(temp)

        # append the word as it is
        else:
            new_string.append(word)

    # join the words of new_string to form a string
    temp_str = ' '.join(new_string)
    return temp_str


# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


# remove whitespace from text
def remove_whitespace(text):
    return " ".join(text.split())


# Replace contractions in string of text
def replace_contractions(text):
    return contractions.fix(text)


# Remove URLs from a sample string
def remove_URL(sample):
    return re.sub(r"http\S+", "", sample)


# remove stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text


# Check if each character in string is ASCII
# The first 128 unicode code points represent the ASCII characters.
def remove_non_ascii(text):
    return ''.join(char for char in text if ord(char) < 128)


# Lemmatize verbs in list of tokenized words
def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []

    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)

    return lemmas


def normalize(words):
    words = remove_URL(words)
    words = remove_non_ascii(words)
    words = text_lowercase(words)
    words = remove_punctuation(words)
    words = remove_whitespace(words)
    words = replace_contractions(words)
    words = remove_numbers(words)

    return words


# Preprocess text
def preprocess_text(corpus):
    # Normalize all the text
    text = normalize(corpus)

    # Remove the stop words
    text = remove_stopwords(text)
    #
    # # Lemmatize and tokenize the text
    text = lemmatize_verbs(text)

    return text


