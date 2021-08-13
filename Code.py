# Importing the Libraries #

import pandas as pd

import numpy as np

import seaborn as sns



import warnings

warnings.filterwarnings( 'ignore' )





# Loading the Dataset #

Dataset = pd.read_csv( "C:/Users/Vaibhavi Nayak/Desktop/Project/4. Breast Cancer Prediction/Dataset.csv" )





# Analyzing the Dataset #


# print( Dataset.head() )


# print( Dataset.info() )


# print( Dataset.describe() )


# print( Dataset.shape )


# print( Dataset.columns )





# Cleaning the dataset #


# Handling Missing Data #

#print( Dataset.isnull().sum().sort_values( ascending = False ) )

#Dataset = Dataset.dropna( thresh = 0.70 *len( Dataset ) , axis = 1 )

#print( Dataset.isnull().sum() )


# Correlation Matrix #

#Correlation_Matrix = Dataset.corr().round( 2 )

#print( Correlation_Matrix )


# Heatmap #

#sns.heatmap( data = Correlation_Matrix ) 





# Feature Selection #


# Converting the Categorical Data to Numerical Data #

from sklearn.preprocessing import LabelEncoder

for column in Dataset.columns:
    
  if ( Dataset[ column ].dtype == np.int64 ) or ( Dataset[ column ].dtype == np.float64 ):

    continue

  Dataset[ column ] = LabelEncoder().fit_transform( Dataset[ column ] )



# Removing Unecessary Features #

y = Dataset[ 'diagnosis' ]

L = [ 'Unnamed: 32' , 'id' , 'diagnosis' ]

x = Dataset.drop( L , axis = 1 )



# Splitting the Dataset #

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





# Model Selection #


# Support Vector Machine Algorithm #





# Training Testing the Model #

from sklearn.svm import SVC

Model = SVC()

Model.fit( X_Training_Dataset , Y_Training_Dataset )

Prediction = Model.predict( X_Testing_Dataset )





# Conclusion #

from sklearn.metrics import accuracy_score

Accuracy = accuracy_score( Y_Testing_Dataset , Prediction )

print( Accuracy )










