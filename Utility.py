def ShowNullDataInfo(injuryCauseData):
    print("No. of columns containing null values")
    print(len(injuryCauseData.columns[injuryCauseData.isna().any()]))

    print("No. of columns not containing null values")
    print(len(injuryCauseData.columns[injuryCauseData.notna().all()]))

    print("Total no. of columns in the dataframe")
    print(len(injuryCauseData.columns))

    print("Total no. of rows in the dataframe")
    print(len(injuryCauseData))


def GetTotalVocabularyForInjuryDescription(injury_data):
    injury_data["Number of Words"] = injury_data["InjuryDesc"].apply(lambda n: len(n.split()))
    print(injury_data.head())

    total_words = 0
    for wordCount in injury_data["Number of Words"]:
        total_words += wordCount

    print("Total words: " + str(total_words))
