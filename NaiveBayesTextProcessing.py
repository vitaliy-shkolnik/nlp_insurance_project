#################################################################################################
#                                                                                               #
# This file deals with the processing of the text that I want to do specifically while using    #
# the Naive Bayes algorithm.  The processing for this algorithm is not necessarily what is      #
# required for others, which is why it is separated out.  A future piece of work would be       #
# find elements that are the same across all and make one generic file for all and specific     #
# ones after that.                                                                              #
#                                                                                               #
#################################################################################################


# Remove a row if either column is null
def remove_null_rows_from_injury_dataset(injury_data):
    return injury_data[(injury_data.InjuryDesc.notnull()) & (injury_data.InjuryCauseDesc.notnull())]
