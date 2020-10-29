CS 4372 Assignment 4
James Hooper ~ NETID: jah171230
Hritik Panchasara ~ NETID: hhp160130
------------------------------------------------------------------------------------------------------------------------------------
- For this assignment we utilized PyCharm
- Python Version: 3.7
- Imported Packages:
	- Pandas
	- Numpy
	- Sklearn
- The data is held in a github repository. Due to this the link is embedded in the code to make things simpler.
- REMEMBER: This code does not require arguments. 
- Just run it with the proper packages/python-version and the url link to the data should already be processed.
------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS & PARAMETERS
if __name__ == '__main__':
~ Calls the pre-process function.
~ Runs the model function 5 times for each trial (5 trials).
~ Prints out the parameters chosen and the results.
~ Finally prints out the query from part 8 of the pdf asking for the average sentiment of each airline.

def process_data():
~ Pre-processses the data.
~ There is a paramter of whether to print out certain text that confirms the data is pre-processed correctly.
~ Gets the proper 3 columns.
~ Converts to lower case
~ Utilizes count vectorize & tfidftransformer
~ Encodes the airline sentiment to a numerical value.

def model(train_dataset, validation_dataset, test_dataset, labels_of_classes):
~ Splits the data.
~ Creates the MultinomialNB model
~ Makes a prediction based on the model.
~ Prints the confusion matrix & a set of 25 prediceted values alongside a set of 25 y_test actual values for comparison.
------------------------------------------------------------------------------------------------------------------------------------
- The Report has the final analysis of the results and the answer to which airline has the highest positive sentiment.