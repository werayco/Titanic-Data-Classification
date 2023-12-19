import pandas as pd
import pickle as pkl
import numpy
import numpy as np

pd.set_option("display.max_rows",None)
# loading and instantiating the binary files
test_data = pd.read_csv("test.csv")
with open("transformer.pkl","rb") as processor:
    ColumnTrans=pkl.load(processor)
with open("model.pkl","rb") as model:
    theModel=pkl.load(model)


test_features=test_data[['Pclass','Age','SibSp','Parch','Fare','Sex','Embarked']]
transformed_data = ColumnTrans.transform(test_features)
pred=theModel.predict(transformed_data)

# now lets add the predicted data to the test dataframe
test_data["Survived"]=pred
test_data["Survived"]=test_data["Survived"].astype("int")
important=test_data[["Survived","PassengerId"]]
important.to_csv("SurviedTest001.csv",header=True,index=False)
survived = test_data["Survived"]
survived.to_csv("SurviedTest01.csv",header=True,index=False)
print(test_data)