<H3>ENTER YOUR NAME       : Rishab p Doshi</H3>
<H3>ENTER YOUR REGISTER NO:212224240134</H3>
<H3>EX.NO:1</H3>
<H3>DATE</H3>
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

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))


```

## OUTPUT:

### Dataset:

<img width="1277" height="468" alt="image" src="https://github.com/user-attachments/assets/b21fcf7a-f057-4c2e-accd-d4e61b46a4d8" />

### X Values:

<img width="726" height="178" alt="image" src="https://github.com/user-attachments/assets/e2ffd970-6256-43da-97af-d1ddde17c21c" />

### Y Values:

<img width="670" height="60" alt="image" src="https://github.com/user-attachments/assets/99b54178-6c44-43bf-94b0-c8a2d50aacf9" />

### Null Values:

<img width="315" height="339" alt="image" src="https://github.com/user-attachments/assets/b67c1c75-ef32-4107-b885-461f45ec9b55" />

### Duplicated Values:

<img width="372" height="284" alt="image" src="https://github.com/user-attachments/assets/8f1a8de2-5747-4525-a13d-c1493c94ecb8" />

### Description:

<img width="1269" height="346" alt="image" src="https://github.com/user-attachments/assets/073f08d6-98d3-488c-86b5-dae0177541de" />

### Normalized Dataset:

<img width="799" height="599" alt="image" src="https://github.com/user-attachments/assets/85b6c9c4-d4f2-4548-9c6e-8f36c4f48a9d" />

### Training Data:

<img width="766" height="172" alt="image" src="https://github.com/user-attachments/assets/afa3e758-458b-4081-ae46-6b680aa3ab6b" />

### Testing Data:

<img width="819" height="175" alt="image" src="https://github.com/user-attachments/assets/3b5258fb-0bd0-4ab0-bbc4-3e804cc94f2a" />

<img width="222" height="33" alt="image" src="https://github.com/user-attachments/assets/0d9aad6b-e24e-4b7b-b901-f4f3c84a8895" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


