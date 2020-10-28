# CS 4372 Assignment 4
# Twitter Airline Sentiment Anaylsis
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

# Pre-process data
def process_data(show_prints):
    # Read in csv file into pandas dataframe
    url = "https://raw.githubusercontent.com/jamesH-48/Twitter-Airline-Sentiment-Anaylsis/main/Tweets.csv"
    df = pd.read_csv(url, sep=',')
    # Read in the 3 specified columns
    # Airline Sentiment, Airline, and Text
    df = df[['airline_sentiment','airline','text']]
    if show_prints:
        print("Dataframe Columns:")
        for cols in df.columns:
            print('- ', cols)

    # spacing
    if show_prints:
        print()

        # Convert text to lowercase
        print("Airline Sentiment to lower case")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Before:")
        print(df[['airline_sentiment']].head())
    df['airline_sentiment'] = df.airline_sentiment.map(lambda x: x.lower())
    if show_prints:
        print("-------------------------------------------------")
        print("After:")
        print(df[['airline_sentiment']].head())
        print("-------------------------------------------------")
        print()

        print("Airline to lower case")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Before:")
        print(df[['airline']].head())
    df['airline'] = df.airline.map(lambda x: x.lower())
    if show_prints:
        print("-------------------------------------------------")
        print("After:")
        print(df[['airline']].head())
        print("-------------------------------------------------")
        print()

        print("Text to lower case")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Before:")
        print(df[['text']].head())
    df['text'] = df.text.map(lambda x: x.lower())
    if show_prints:
        print("-------------------------------------------------")
        print("After:")
        print(df[['text']].head())
        print("-------------------------------------------------")
        print()

    # Transform the text data using CountVectorize and TfidTransformer
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(df['text'])
    transformer = TfidfTransformer().fit(counts)
    counts = transformer.transform(counts)
    if show_prints:
        print(type(counts))
        print(counts.get_shape())

    # Convert the airline_sentiment from categorical to numerical values
        print()
        print("Encode Airline Sentiment")
        print("Original:")
        print(df[['airline_sentiment']].head(10))
    le = preprocessing.LabelEncoder()
    df['airline_sentiment'] = le.fit_transform(df['airline_sentiment'])
    if show_prints:
        print("Encoded:")
        print(df[['airline_sentiment']].head(10))

    return counts, df

# Create model for evaluation of data
def model(counts, df, test_size, alpha, fit_prior):
    X_train, X_test, y_train, y_test = train_test_split(counts, df['airline_sentiment'], test_size=test_size)
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print()

    model = MultinomialNB(alpha=alpha, fit_prior=fit_prior).fit(X_train, y_train)
    predicted = model.predict(X_test)

    print("Model Prediction Result:")
    print(np.mean(predicted == y_test), "\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predicted), "\n")

    print("Predicted")
    print(predicted[:25])
    print("\nY_test")
    y_test_array = y_test.to_numpy()
    print(y_test_array[:25])

# Main
if __name__ == "__main__":
    #print(sys.argv)
    # Set up sys.argv[0] to have url and to only accept sys.argv size of 1

    # Prints are basic confirmations of the data for the coder/user
    # There will always be the constant print out of the results of the model
    show_preprocess_prints = False
    counts, df = process_data(show_preprocess_prints)

    '''
        Repeat this 5 times with different parameters each.
    '''
    # Set parameters for different model results
    print("------------------------------------------------------------------------------------")
    print("Trial 1")
    print("------------------------------------------------------------------------------------")
    test_size = .1
    alpha = 1
    fit_prior = True
    print("Test Size: ", test_size)
    print("Alpha: ", alpha)
    print("Fit Prior: ", fit_prior)
    model(counts, df, test_size, alpha, fit_prior)

    print("------------------------------------------------------------------------------------")
    print("Trial 2")
    print("------------------------------------------------------------------------------------")
    test_size = .1
    alpha = 1
    fit_prior = False
    print("Test Size: ", test_size)
    print("Alpha: ", alpha)
    print("Fit Prior: ", fit_prior)
    model(counts, df, test_size, alpha, fit_prior)

    print("------------------------------------------------------------------------------------")
    print("Trial 3")
    print("------------------------------------------------------------------------------------")
    test_size = .2
    alpha = 1
    fit_prior = True
    print("Test Size: ", test_size)
    print("Alpha: ", alpha)
    print("Fit Prior: ", fit_prior)
    model(counts, df, test_size, alpha, fit_prior)

    print("------------------------------------------------------------------------------------")
    print("Trial 4")
    print("------------------------------------------------------------------------------------")
    test_size = .2
    alpha = 1
    fit_prior = False
    print("Test Size: ", test_size)
    print("Alpha: ", alpha)
    print("Fit Prior: ", fit_prior)
    model(counts, df, test_size, alpha, fit_prior)

    print("------------------------------------------------------------------------------------")
    print("Trial 5")
    print("------------------------------------------------------------------------------------")
    test_size = .3
    alpha = 1
    fit_prior = True
    print("Test Size: ", test_size)
    print("Alpha: ", alpha)
    print("Fit Prior: ", fit_prior)
    model(counts, df, test_size, alpha, fit_prior)
    print("------------------------------------------------------------------------------------")

    # Using the numeric value of airline_sentiment, output the average sentiment of each airline
    # and report which airline has the highest positive sentiment.
    # The closer the value is to 0 the more negative the sentiment
    # A netural sentiment is a value of 1
    # The closer the value is to 2 the more positive the sentiment
    print("\n------------------------------------------")
    print("Average Sentiment of Each Airline")
    print("Negative = 0")
    print("Neutral = 1")
    print("Positive = 2")
    print("------------------------------------------")
    # Get the unique values from the dataframe of all the airlines
    airlines = df.airline.unique()
    airline_sentiments = []
    # Get the average sentiment of each airline
    for airline in airlines:
        AL_cur = df['airline_sentiment'].loc[df['airline'] == airline]
        airline_sentiments.append(AL_cur.mean())

    airline_sentiments = np.array(airline_sentiments)
    AS = {'airline': airlines, 'avg_sentiment': airline_sentiments}
    AS_df = pd.DataFrame(data=AS)
    print(AS_df)
