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

# Reading the data from pickle file where it has already been processed
injuryDataFrame = pd.read_pickle("Data/injury_descriptions_processed.pkl")

# Now that the data has been processed, split it.
X_train, X_test, y_train, y_test = train_test_split(injuryDataFrame['processed_desc'],
                                                    injuryDataFrame['InjuryCauseDesc'],
                                                    random_state=0)

vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, decode_error="ignore", ngram_range=(1, 3))
vectorizer.fit(X_train)

cls = MultinomialNB()

# transform the list of text to tf-idf before passing it to the model
cls.fit(vectorizer.transform(X_train), y_train)

y_pred = cls.predict(vectorizer.transform(X_test))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
