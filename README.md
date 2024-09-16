<H3>ENTER YOUR NAME:SANTHA RAMANATH M</H3>
<H3>ENTER YOUR REGISTER NO: 212223220097</H3>
<H3>EX. NO.1</H3>
<H3>DATE:16/09/2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('/content/Churn_Modelling.csv')
print(df.head())

X=df.iloc[:,:-1].values
print(X)

y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())

df.duplicated()

df.describe()

df = df.drop(['Surname', 'Geography','Gender'], axis=1)
df.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(X_train)
print(len(X_train))

print(X_test)
print(len(X_test))
```


## OUTPUT:
## DATASET:

![359739596-63cfc27c-6c61-48ca-a78f-823629c3d564](https://github.com/user-attachments/assets/aaeed8ab-7e41-44a4-b0cc-4137f0701c8d)


## X VALUES:

![359739617-cb69b34e-382c-4865-ae99-cc56070ba46e](https://github.com/user-attachments/assets/d4cd5173-7efc-4b8c-a259-5c91d4d9bd55)


## Y VALUES:

![359739642-91285301-aed9-4455-a899-994d73ad0950](https://github.com/user-attachments/assets/1da4926e-e957-47f8-85fd-b6494663658d)

## NULL VALUES:

![359739663-741f94bf-c1c1-477f-91cb-5388f04878a8](https://github.com/user-attachments/assets/ad197812-ea78-4c87-8896-1b7622ed5d9b)



## DUPLICATED VALUES:

![359739682-4b4c2a61-217d-410b-90d0-84f08ad5282f](https://github.com/user-attachments/assets/b4f2177d-bd7b-4b1b-a1f7-f0982642aced)


## DESCRIPTION:

![359739698-15cabb08-6f19-4002-a8ee-0932040b6afd](https://github.com/user-attachments/assets/7b75c288-1382-49c6-8c0a-0a3424093258)


## NORMALIZED DATASET:

![359740105-9020fe99-6e44-4871-ab06-4fb9279b43af](https://github.com/user-attachments/assets/43feb7a2-7fc5-43a9-988d-d4ab20adb981)


## TRAINING DATASET:

![359740116-46600a57-4529-4c57-a68e-1461af2f0cff](https://github.com/user-attachments/assets/72bff819-8ca9-4ed5-a9c2-db1fa50a07cb)


## TESTING DATASET:

![359740124-7353c8e4-d2af-4bd5-aa78-4d9ba81fb5e3](https://github.com/user-attachments/assets/b5e75739-b537-4aaf-b7ae-f51194c5c84e)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


