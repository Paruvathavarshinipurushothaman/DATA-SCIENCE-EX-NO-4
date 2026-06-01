# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
df=pd.read_csv("bmi.csv")
df
```
<img width="539" height="650" alt="image" src="https://github.com/user-attachments/assets/ba33c390-72e6-4eed-a83d-edffddff6198" />

```
df.head()
```

<img width="493" height="331" alt="image" src="https://github.com/user-attachments/assets/3eb2376d-50ae-43eb-a765-530184e44dcb" />

```
df.dropna()
```

<img width="455" height="655" alt="image" src="https://github.com/user-attachments/assets/b43b190d-0b18-4d69-b7b0-bc51d4d288d2" />

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```

<img width="273" height="67" alt="image" src="https://github.com/user-attachments/assets/a49e9aac-f488-45e0-b009-95fbc7f056f8" />

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

<img width="502" height="580" alt="image" src="https://github.com/user-attachments/assets/63f19bed-1ca2-4b4d-a7a6-c572bb3a695d" />

```
df1=pd.read_csv("bmi.csv")
df2=pd.read_csv("bmi.csv")
df3=pd.read_csv("bmi.csv")
df4=pd.read_csv("bmi.csv")
df5=pd.read_csv("bmi.csv")
df1
```

<img width="478" height="644" alt="image" src="https://github.com/user-attachments/assets/a0779a99-a6d5-48b0-ae03-39adf1ed55fd" />

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df.head(10)
```

<img width="478" height="555" alt="image" src="https://github.com/user-attachments/assets/70c704e0-45b1-4971-96e4-dfd570317850" />

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```

<img width="493" height="640" alt="image" src="https://github.com/user-attachments/assets/d2dd4686-b677-4af1-9a23-8aeed31831eb" />

```
from sklearn.preprocessing import MaxAbsScaler
max1=MaxAbsScaler()
df3[['Height','Weight']]=max1.fit_transform(df3[['Height','Weight']])
df3
```

<img width="491" height="648" alt="image" src="https://github.com/user-attachments/assets/36668313-730d-4002-bf35-8d1fac60361a" />

```
from sklearn.preprocessing import RobustScaler
roub=RobustScaler()
df4[['Height','Weight']]=roub.fit_transform(df4[['Height','Weight']])
df4
```

<img width="561" height="650" alt="image" src="https://github.com/user-attachments/assets/126d4da8-04a4-4c25-adf4-d82463509f68" />

```
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif
from sklearn.feature_selection import chi2
data=pd.read_csv("income(1) (1).csv")
data
```

<img width="1027" height="309" alt="image" src="https://github.com/user-attachments/assets/a2879a28-d9fc-4e80-bc23-42d4f7620696" />

```
data1=pd.read_csv('/content/titanic_dataset (1).csv')
data1
```

<img width="1043" height="409" alt="image" src="https://github.com/user-attachments/assets/c460d399-9635-46b0-9154-de9a688233e1" />

```
data1=data1.dropna()
x=data1.drop(['Survived','Name','Ticket'],axis=1)
y=data1['Survived']
data1['Sex']=data1['Sex'].astype('category')
data1['Cabin']=data1['Cabin'].astype('category')
data1['Embarked']=data1['Embarked'].astype('category')

data1['Sex']=data1['Sex'].cat.codes
data1['Cabin']=data1['Cabin'].cat.codes
data1['Embarked']=data1['Embarked'].cat.codes

data1
```

<img width="1039" height="403" alt="image" src="https://github.com/user-attachments/assets/883fa69f-77ac-44aa-9405-022ffb8db5b1" />

```
k=5
selector=SelectKBest(score_func=chi2,k=k)
x=pd.get_dummies(x)
x_new=selector.fit_transform(x,y)
x_encoded=pd.get_dummies(x)
selector=SelectKBest(score_func=chi2,k=5)
x_new=selector.fit_transform(x_encoded,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="1037" height="84" alt="image" src="https://github.com/user-attachments/assets/5a002390-d130-46c5-88e4-39927f5b9b22" />

```
selector=SelectKBest(score_func=f_regression,k=5)
x_new=selector.fit_transform(x_encoded,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="1037" height="74" alt="image" src="https://github.com/user-attachments/assets/cbd726da-9367-4fdc-864b-ba6584025a91" />

```
selector=SelectKBest(score_func=mutual_info_classif,k=5)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="1035" height="86" alt="image" src="https://github.com/user-attachments/assets/85de1514-8b25-474b-95ea-ceb011592c7a" />

```
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
x=pd.get_dummies(x)
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)
```

<img width="1035" height="154" alt="image" src="https://github.com/user-attachments/assets/ff89fde9-d778-49f2-b466-811ae47c167b" />

```
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_selection=model.feature_importances_
threshold=0.1
selected_features=x.columns[feature_selection>threshold]
print("Selected Features:")
print(selected_features)
```

<img width="1037" height="105" alt="image" src="https://github.com/user-attachments/assets/02c68a65-8904-4017-b079-00159457d767" />

```
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importance=model.feature_importances_
threshold=0.15
selected_features=x.columns[feature_importance>threshold]
print("Selected Features:")
print(selected_features)
```

<img width="419" height="71" alt="image" src="https://github.com/user-attachments/assets/6bf559b5-6e9c-4211-8e86-d313142d8ede" />


   
# RESULT:
       Thus the feature selection and feature scaling has been used on the given dataset and executed successfully.
