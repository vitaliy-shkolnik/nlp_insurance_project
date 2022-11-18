#################################################################################################
#                                                                                               #
# This file will contain the construction of a Naive Bayes algorithm using the Sklearn          #
# library.  We will be using worker's compensation injury description data to predict the       #
# cause of injury.                                                                              #
#                                                                                               #
#################################################################################################
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from NaiveBayesTextProcessing import remove_null_rows_from_injury_dataset
from TextPreprocessing import normalize, lemmatize_sentence

# Reading the data
injuryDataFrame = pd.read_csv("Data/InjuryCauseTopThirteen.csv", sep=',', encoding='latin-1',
                              usecols=lambda col: col not in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

# Quick exam of the data
print(injuryDataFrame.head())

# What does the text look like?
print("\n\nSome example injury text:")
print(injuryDataFrame["InjuryDesc"].to_list()[:5])

# Remove the rows that have null
injuryDataFrame = remove_null_rows_from_injury_dataset(injuryDataFrame)

# Normalize the data
injuryDataFrame['processed_desc'] = [normalize(x) for x in injuryDataFrame['InjuryDesc']]

# What does the text look like now?
print("\n\nSome example injury text having been normalized:")
print(injuryDataFrame["processed_desc"].to_list()[:5])

# Lemmatize the sentence.  This takes a while to run on a standard machine
injuryDataFrame['processed_desc'] = [lemmatize_sentence(x) for x in injuryDataFrame['processed_desc']]

# What does the text look like now?
print("\n\nSome example injury text having been lemmatized:")
print(injuryDataFrame["processed_desc"].to_list()[:5])

# Now that the data has been processed, split it.
X_train, X_test, y_train, y_test = train_test_split(injuryDataFrame['processed_desc'],
                                                    injuryDataFrame['InjuryCauseDesc'],
                                                    random_state=0)

vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, decode_error="ignore", ngram_range=(2, 3))
vectorizer.fit(X_train)

cls = MultinomialNB()

# transform the list of text to tf-idf before passing it to the model
cls.fit(vectorizer.transform(X_train), y_train)

y_pred = cls.predict(vectorizer.transform(X_test))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
