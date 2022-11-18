#################################################################################################
#                                                                                               #
# This file will contain the construction of a Naive Bayes algorithm using the Sklearn          #
# library.  We will be using worker's compensation injury description data to predict the       #
# cause of injury.                                                                              #
#                                                                                               #
#################################################################################################
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Reading the data from pickle file where it has already been processed
injuryDataFrame = pd.read_pickle("Data/injury_descriptions_processed.pkl")

# Removed troublesome category here.  For some reason this one is a very low predictor.
injuryDataFrame = injuryDataFrame[
    ~injuryDataFrame.InjuryCauseDesc.str.contains("Bodily reaction and exertion, unspecified")]
injuryDataFrame = injuryDataFrame[~injuryDataFrame.InjuryCauseDesc.str.contains("Bodily reaction, n.e.c.")]

# labels for heat map/confusion matrix

category_id_df = injuryDataFrame[['InjuryCauseDesc']].drop_duplicates().sort_values('InjuryCauseDesc')
# Now that the data has been processed, split it.
X_train, X_test, y_train, y_test = train_test_split(injuryDataFrame['processed_desc'],
                                                    injuryDataFrame['InjuryCauseDesc'],
                                                    random_state=0)

vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, decode_error="ignore", ngram_range=(1, 3))
vectorizer.fit(X_train)

cls = MultinomialNB()

# transform the list of text to tf-idf before passing it to the model
cls.fit(vectorizer.transform(X_train), y_train)

y_pred = cls.predict(vectorizer.transform(X_test))

# Display metrics
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Display heatmap confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(15, 13))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.InjuryCauseDesc.values, yticklabels=category_id_df.InjuryCauseDesc.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
