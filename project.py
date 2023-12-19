import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import pickle as pkl
from sklearn.model_selection import KFold,train_test_split

import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import os
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
import sys
import numpy as np
from modelselect import best_model

from sklearn.preprocessing import OneHotEncoder,StandardScaler

trainData_ValidData=pd.read_csv("train.csv")

kf = KFold(n_splits=2,shuffle=True,random_state=39)

for trainIndex,ValidIndex in kf.split(trainData_ValidData):
    trainData,ValidData = trainData_ValidData.iloc[trainIndex],trainData_ValidData.iloc[ValidIndex]


trainData.to_csv('TrainedValid\TrainData.csv',header=True,index=True)
ValidData.to_csv('TrainedValid\ValidData.csv',header=True,index=True)

# imp_data=trainData_ValidData[['Pclass','Age','SibSp','Parch','Fare','Sex','Cabin','Embarked']]


















# print(data.isna().sum())

def datacleaning():
    num_columns=['Pclass','Age','SibSp','Parch','Fare']
    cat_columns=['Sex','Embarked']
    target=['Survived']
    num_pip=Pipeline(steps=[('imputer',SimpleImputer(strategy='mean')),('standardizer',StandardScaler(with_mean=False))])
    cat_pip=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder())])
    transformer=ColumnTransformer([('cat',cat_pip,cat_columns),('num',num_pip,num_columns)])
    return transformer

# reading both Validation and train Data
Train_Data=pd.read_csv('TrainedValid\TrainData.csv')
Valid_Data=pd.read_csv('TrainedValid\ValidData.csv')

# feature engineering
Train_Data_features = Train_Data[['Pclass','Age','SibSp','Parch','Fare','Sex','Embarked']]
Valid_Data_features=Valid_Data[['Pclass','Age','SibSp','Parch','Fare','Sex','Embarked']]







# target engineering
Train_Data_target = Train_Data['Survived']
Valid_Data_target = Valid_Data['Survived']
# calling the datacleaning function
datacleaning_obj=datacleaning()

# fitting the preprocessor to the respective dataframes
cleanedTrainData=datacleaning_obj.fit_transform(Train_Data_features)
cleanedValidData=datacleaning_obj.transform(Valid_Data_features)

#join the cleaned dataframes to their respective targets
train_array = np.c_[cleanedTrainData,np.array(Train_Data_target)]
valid_array = np.c_[cleanedValidData,np.array(Valid_Data_target)]






# MODEL ENGINEERING

models={"Logistic_Regression":LogisticRegression(),
        "K_Neighbour Classifier":KNeighborsClassifier(n_neighbors=3),"Random_forest":RandomForestClassifier(),"SVC":SVC(),}

x_train,y_train,x_valid,y_valid=train_array[:,:-1],train_array[:,-1],valid_array[:,:-1],valid_array[:,-1]



params = {
    "Logistic_Regression": {},
    "K_Neighbour Classifier": {'n_neighbors': [5, 7, 9, 11]},
    "Random_forest": {
        'n_estimators': [100,150, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'bootstrap': [True, False]},
    "SVC":{'C': [0.1, 1, 10,0.6],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']}
    
}


model_scores:dict=best_model(x_train,y_train=y_train,x_test=x_valid,y_test=y_valid,models=models,params=params)



 # best model score from the dict
best_model_score=max(sorted(list(model_scores.values())))

best_model_name=list(model_scores.keys())[list(model_scores.values()).index(best_model_score)]
# the above code will return a string
print(f"the best model is {best_model_name}")

best_m=models[best_model_name]
y_pred=best_m.predict(x_train)
score=accuracy_score(y_train,y_pred)
conf=confusion_matrix(y_train,y_pred)
precision_scor=precision_score(y_train,y_pred)
print(score)
print(conf)
print(precision_scor)

with open("model.pkl","wb") as file:
    pkl.dump(best_m,file)
with open("transformer.pkl","wb") as file2:
    pkl.dump(datacleaning_obj,file2)






    
