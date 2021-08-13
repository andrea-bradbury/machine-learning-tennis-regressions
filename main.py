#Using machine learning to determine a number of factors related to an ATP tennis player's ranking.
#Data from the top 1500 ranked players in the ATP over the span of 2009 to 2017 are provided in file.
#The statistics recorded for each player in each year include service game (offensive) statistics, return game (defensive) statistics and outcomes.
#Using pandas, matplotlib and sklearn to plot results
#An exercise as part of Codecademy's Machine Learning with Python course

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data
players = pd.read_csv('tennis_stats.csv')
print(players.head())
print(players.columns)
print(players.describe())

# exploratory analysis
print(players.corr())
plt.scatter(players['FirstServeReturnPointsWon'],players['Winnings'])
plt.title('First Serve Return Points Won vs Winnings')
plt.xlabel('First Serve Return Points Won')
plt.ylabel('Winnings')
plt.show()
plt.clf()

plt.scatter(players['BreakPointsOpportunities'],players['Winnings'])
plt.title('Break Points Opportunities vs Winnings')
plt.xlabel('Break Points Opportunities')
plt.ylabel('Winnings')
plt.show()
plt.clf()

plt.scatter(players['BreakPointsSaved'],players['Winnings'])
plt.title('Break Points Saved vs Winnings')
plt.xlabel('Break Points Saved')
plt.ylabel('Winnings')
plt.show()
plt.clf()

plt.scatter(players['TotalPointsWon'],players['Ranking'])
plt.title('Total Points Won vs Ranking')
plt.xlabel('Total PointsWon')
plt.ylabel('Ranking')
plt.show()
plt.clf()

plt.scatter(players['TotalServicePointsWon'],players['Wins'])
plt.title('Total Service Points Won vs Wins')
plt.xlabel('Total Service Points Won')
plt.ylabel('Wins')
plt.show()
plt.clf()

## single feature linear regression (FirstServeReturnPointsWon)

# select features and value to predict
features = players[['FirstServeReturnPointsWon']]
winnings = players[['Winnings']]

# train, test, split the data
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

# create and train model on training data
model = LinearRegression()
model.fit(features_train,winnings_train)

# score model on test data
print('Predicting Winnings with First Serve Return Points Won Test Score:', model.score(features_test,winnings_test))

# make predictions with model
winnings_prediction = model.predict(features_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - 1 Feature')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()

## single feature linear regression (BreakPointsOpportunities)

# select features and value to predict
features = players[['BreakPointsOpportunities']]
winnings = players[['Winnings']]

# train, test, split the data
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

# create and train model on training data
model = LinearRegression()
model.fit(features_train,winnings_train)

# score model on test data
print('Predicting Winnings with Break Points Opportunities Test Score:', model.score(features_test,winnings_test))

# make predictions with model
winnings_prediction = model.predict(features_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - 1 Feature')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()

## two feature linear regression

# select features and value to predict
features = players[['BreakPointsOpportunities','FirstServeReturnPointsWon']]
winnings = players[['Winnings']]

# train, test, split the data
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

# create and train model on training data
model = LinearRegression()
model.fit(features_train,winnings_train)

# score model on test data
print('Predicting Winnings with 2 Features Test Score:', model.score(features_test,winnings_test))

# make predictions with model
winnings_prediction = model.predict(features_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - 2 Features')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()

## multiple features linear regression

# select features and value to predict
features = players[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon','SecondServePointsWon','SecondServeReturnPointsWon','Aces','BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities','BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon','TotalServicePointsWon']]
winnings = players[['Winnings']]

# train, test, split the data
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

# create and train model on training data
model = LinearRegression()
model.fit(features_train,winnings_train)

# score model on test data
print('Predicting Winnings with Multiple Features Test Score:', model.score(features_test,winnings_test))

# make predictions with model
winnings_prediction = model.predict(features_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - Multiple Features')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()


# Summary
print('Predicting Winnings with Multiple Features has the highest test score and is the best model for accuracy.')