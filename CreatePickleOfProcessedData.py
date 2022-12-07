import pandas as pd
from NaiveBayesTextProcessing import remove_null_rows_from_injury_dataset
from TextPreprocessing import lemmatize_sentence, normalize

# Reading the data
injuryDataFrame = pd.read_csv("Data/InjuryCauseTopThirteen_cd_clean_final.csv", sep=',', encoding='latin-1',
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

injuryDataFrame.to_pickle("Data/injury_descriptions_processed_cleaned.pkl")
