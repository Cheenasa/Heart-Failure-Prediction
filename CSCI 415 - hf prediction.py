#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


data = pd.read_csv(r'C:\Users\SystemNova\Documents\NasaFolder\Datasets\heart.csv') #Read data from CSV into the notebook
df = data.copy() #Make a copy of the dataset to avoid changes to the original dataset.


# In[3]:


print(f' There are {df.shape[0]} row and {df.shape[1]} and columns')


# In[4]:


df.head(10) #First 10 rows of the dataset.


# In[5]:


# Plot
# 1. Male vs. Female frequency
sex_counts = df.Sex.value_counts().values
sex_labels = df.Sex.value_counts().index

plt.pie(
    x=sex_counts,
    labels=sex_labels,
    autopct='%1.1f%%',
    shadow=True,
)

plt.title("Frequency of Male And Female In Dataset");


# In[6]:


# Plot
# 1. Male vs. Female frequency
HrtDis_counts = df.HeartDisease.value_counts().values
HrtDis_labels = df.HeartDisease.value_counts().index

plt.pie(
    x=HrtDis_counts,
    labels=HrtDis_labels,
    autopct='%1.1f%%',
    shadow=True,
)

plt.title("Frequency of those that have heart disease or not In Dataset");


# In[7]:


df.info() #Information about the dataset


# In[8]:


df.describe(include='all').T  #Gives the statistical description of the dataset.


# * From the description:
#     * The range for age is 28 - 77 years. These can be binned for visualization.
#     * There are 2 unique values in sex(Male and Female). No preprocessing needed.
#     * Cholesterol has some zero values. These are missing values and should be treated as such.
#     * 

# In[9]:


sns.pairplot(data=df, hue='HeartDisease', kind='kde');


# * From the pairplot:
#     * The lower your MaxHR, the likely you will have heart disease.
#     * The higher your age, fastingBS, and Oldpeak value, the more likely you will have the disease.
#     * RestingBS have no significant impact.
#     * MaxHR and age are positively correlated.
#     * Recods with low cholesterol are seen to have the risk of heart diseases esp cholesterol levels of 0. According to research, it is very rare for an individual to have a cholesterol level of 0. We will therefore treat these as missing items.
#     * Further reading: https://s4be.cochrane.org/blog/2018/07/02/cholesterol-and-heart-disease-whats-the-evidence/

# In[10]:


df[df['Cholesterol']==0]


# * There are 172 records with cholesterol = 0 values. 

# In[11]:


df[df['Cholesterol']>500]


# * These data points are not too far from the expected norm. Therefore we wont treat them.

# In[12]:


df['Cholesterol'].replace(0, np.nan, inplace=True)
df.dropna(inplace=True)


# In[13]:


sns.pairplot(data=df, hue='HeartDisease', kind='kde');


# * From the pairplot:
#     * Cholesterol and RestingBS have no significant impact.
#     

# In[14]:


#Showing the histograms of the numerical columns in the dataset.
df.hist(figsize=(14, 14))
plt.show()


# * There are outliers in the RestingBP and Cholesterol columns which needs to be investigated.
# * None of the attruibutes is normally distributed.

# In[15]:


#Discretinazation of the Age column.
df['Age_bins'] = pd.cut(x=df['Age'], bins=[20, 30, 40, 50, 60, 70, 90],
                    labels=['20 to 30', '31 to 40', '41 to 50', '51 to 60',
                            '61 to 70', 'Above 70'])
df


# In[16]:


df[df['Age']>=70]


# In[17]:


df['Age_bins'].value_counts()


# In[18]:


sns.countplot(df, x="Age_bins", hue='HeartDisease')


# For the dataset:
# * Those within the age bracket of 51 to 60 years are more prone to heart diseases. 
# * We can also note that the risk of heart diseases increases with age.

# ## Model Building

# ## Data Preprocessing
# **1. Features to be encoded:**
# - Sex
# - ChestPain Type
# - RestingECG
# - ExerciseAngina
# - ST_Slope
# 
# **2. Drop cholesterol with values of zero.**
# 
# **3. Split the dataset into train and test split.**
# 
# **4. Normalise the dataset.**
# 

# In[19]:


to_be_processed = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# df.replace() method to encode values
data['Sex'] = data.Sex.replace({'M': 0, 'F': 1})
data['ChestPainType'] = data.ChestPainType.replace({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 4})
data['RestingECG'] = data.RestingECG.replace({'Normal': 0, 'ST': 1, 'LVH': 2})
data['ExerciseAngina'] = data.ExerciseAngina.replace({'N': 0, 'Y': 1})
data['ST_Slope'] = data.ST_Slope.replace({'Up': 0, 'Flat': 1, 'Down': 2})


# In[20]:


# Dropping missing valus in Cholesterol.
data['Cholesterol'].replace(0, np.nan, inplace=True)
data.dropna(inplace=True)


# In[21]:


#Normalizing the dataset using standard scaler

data


# In[22]:


X = data.drop('HeartDisease', axis=1)
Y = data['HeartDisease']


# In[23]:


# Splitting data into training and test set


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1, stratify=Y)

print(X_train.shape, X_test.shape)


# In[24]:


# Decision Tree
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)

# Random Forest
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)

# Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)

# K-Nearest Neighbors (KNN)
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)

# Support Vector Machine (SVM)
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)


# In[25]:


def evaluate_model(predictions, y_true):
    accuracy = accuracy_score(y_true, predictions)
    conf_matrix = confusion_matrix(y_true, predictions)
    classification_rep = classification_report(y_true, predictions)

    return accuracy, conf_matrix, classification_rep


# In[26]:


# Evaluate Decision Tree
dt_accuracy, dt_conf_matrix, dt_classification_rep = evaluate_model(dt_predictions, y_test)
print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Confusion Matrix:\n", dt_conf_matrix)
print("Decision Tree Classification Report:\n", dt_classification_rep, '\n')

# Evaluate Random Forest
rf_accuracy, rf_conf_matrix, rf_classification_rep = evaluate_model(rf_predictions, y_test)
print("\nRandom Forest Accuracy:", rf_accuracy)
print("Random Forest Confusion Matrix:\n", rf_conf_matrix)
print("Random Forest Classification Report:\n", rf_classification_rep, '\n')

# Evaluate Naive Bayes
nb_accuracy, nb_conf_matrix, nb_classification_rep = evaluate_model(nb_predictions, y_test)
print("\nNaive Bayes Accuracy:", nb_accuracy)
print("Naive Bayes Confusion Matrix:\n", nb_conf_matrix)
print("Naive Bayes Classification Report:\n", nb_classification_rep, '\n')

# Evaluate K-Nearest Neighbors (KNN)
knn_accuracy, knn_conf_matrix, knn_classification_rep = evaluate_model(knn_predictions, y_test)
print("\nKNN Accuracy:", knn_accuracy)
print("KNN Confusion Matrix:\n", knn_conf_matrix)
print("KNN Classification Report:\n", knn_classification_rep, '\n')

# Evaluate Support Vector Machine (SVM)
svm_accuracy, svm_conf_matrix, svm_classification_rep = evaluate_model(svm_predictions, y_test)
print("\nSVM Accuracy:", svm_accuracy)
print("SVM Confusion Matrix:\n", svm_conf_matrix)
print("SVM Classification Report:\n", svm_classification_rep)


# In[27]:


# Model Performance Dictionary
model_performance = {
    'Decision Tree': dt_accuracy,
    'Random Forest': rf_accuracy,
    'Naive Bayes': nb_accuracy,
    'KNN': knn_accuracy,
    'SVM': svm_accuracy
}


# In[28]:


def compare_accuracies(model_accuracies, colors):
    plt.figure(figsize=(12, 6))
    # model names and accuracies
    models = list(model_accuracies.keys())
    accuracies = list(model_accuracies.values())

    # bar graph
    plt.bar(models, accuracies, color=colors)


    # labels and title
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracies Comparison')
    plt.show()


# In[29]:


compare_accuracies(model_performance, ['Teal', 'Orange', 'Gray', 'Magenta', 'Gold'])


# The models all performed relatively well. However, the random forest model performed better with an accuracy of 87%.

# In[30]:


feature_names = X_test.columns
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize = (12,12))
plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="orange", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# In[31]:


feature_names = X_test.columns
importances = dt_classifier.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize = (12,12))
plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="Gray", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ## Conclusion
# 
# The onset of a heart attack can be very subtle. Models that identify high risk factors of heart attacks can be used to raise awareness and reduce sudden heart attacks and sometimes death from heart related diseases. 
# 
# From the study:
# 
# * ST_slope, Oldpeak, MaxHR, Cholesterol, ExcerciseAngina are the first five major indicators of heartdisease.
# * Also, People with Asympthomatic chest pain, flat ST_Slope, normal RestingECG, and MaxHR between 110 and 145 should be identified and watched closely for any heart related diseases.
# 
