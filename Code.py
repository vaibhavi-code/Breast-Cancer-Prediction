############################################################################################################################################################################################


# Importing the Libraries #

import pandas as pd

import numpy as np


############################################################################################################################################################################################


# Ignoring the Warnings #

import warnings

warnings.filterwarnings( 'ignore' )


############################################################################################################################################################################################


# Importing the Dataset #

Dataset = pd.read_csv( "C:/Users/Vaibhavi Nayak/Desktop/Project/4. Breast Cancer Prediction/Dataset.csv" )


############################################################################################################################################################################################


# Cleaning the dataset #


# 1. Handling Missing Data #

Dataset = Dataset.dropna( thresh = 0.70 *len( Dataset ) , axis = 1 )

Dataset = Dataset.fillna( Dataset.mean() )


# 2. Dropping the Irrelevant Columns #

Columns = [ 'id' ]

Dataset.drop( Columns , axis = 1 , inplace = True )


# 3. Converting the Categorical Data to Numerical Data #

from sklearn.preprocessing import LabelEncoder

for column in Dataset.columns:
    
  if ( Dataset[ column ].dtype == np.int64 ) or ( Dataset[ column ].dtype == np.float64 ):

    continue

  Dataset[ column ] = LabelEncoder().fit_transform( Dataset[ column ] )


############################################################################################################################################################################################


# Feature Selection #


# Splitting the Dataset #

y = Dataset[ 'diagnosis' ]

Columns = [ 'diagnosis' ]

x = Dataset.drop( Columns , axis = 1 )

from sklearn.model_selection import train_test_split

X_Training_Dataset , X_Testing_Dataset , Y_Training_Dataset , Y_Testing_Dataset = train_test_split( x , y , test_size = 0.2 , random_state = 0 )


# Selecting the Features #

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

selected_features = SelectKBest( chi2 , k = 10 ).fit( X_Training_Dataset , Y_Training_Dataset )


# print( 'Score List : ' , selected_features.scores_ )

# print( 'Feature List : ' , X_Training_Dataset.columns )


X_Training_Dataset = selected_features.transform( X_Training_Dataset )

X_Testing_Dataset = selected_features.transform( X_Testing_Dataset )


############################################################################################################################################################################################


# Model Selection #


# Logistic Regression #

from sklearn.linear_model import LogisticRegression

Model_1 = LogisticRegression()

Model_1.fit( X_Training_Dataset , Y_Training_Dataset ) 

Prediction_1 = Model_1.predict( X_Testing_Dataset )


# Support Vector Machine Algorithm #

from sklearn.svm import SVC

Model_2 = SVC()

Model_2.fit( X_Training_Dataset , Y_Training_Dataset )

Prediction_2 = Model_2.predict( X_Testing_Dataset )


############################################################################################################################################################################################


# Conclusion #


from sklearn.metrics import accuracy_score


Accuracy_1 = accuracy_score( Y_Testing_Dataset , Prediction_1 )

print( 'Logistic Regression' , Accuracy_1 )


Accuracy_2 = accuracy_score( Y_Testing_Dataset , Prediction_2 )

print( 'Support Vector Machine' , Accuracy_2 )


############################################################################################################################################################################################









