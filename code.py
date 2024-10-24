#1 Logistic Regression

import pandas as pd 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 

data = sns.load_dataset('iris') 
encoder = LabelEncoder() 
data.species = encoder.fit_transform(data.species) 

x = data[['sepal_length','sepal_width','petal_length','petal_width']] 
y = data['species'] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25) 

model = LogisticRegression() 
model.fit(x_train, y_train) 
prediction = model.predict(x_test) 
accuracy = accuracy_score(prediction, y_test) 
print("the accuracy is:", accuracy)





#2. SVM Classification

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import cross_val_score 

colnames = [
    "sepal_length_in_cm", 
    "sepal_width_in_cm", 
    "petal_length_in_cm", 
    "petal_width_in_cm", 
    "class"
]

dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, names=colnames) 

X = dataset.iloc[:, :-1] 
y = dataset.iloc[:, -1].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) 

classifier = SVC(kernel='linear', random_state=0) 
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test) 

cm = confusion_matrix(y_test, y_pred) 
print(cm) 

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10) 

print("Accuracy: {:.2f} %".format(accuracies.mean() * 100)) 
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))






#3. K Nearest Neighbor Classification

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 

iris = load_iris() 
x = iris.data 
y = iris.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1) 

knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(x_train, y_train) 
print("The accuracy score is:", knn.score(x_test, y_test))







#4. Logistic regression classifier

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 

iris = load_iris() 

x = iris.data 
y = iris.target 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1) 

lr = LogisticRegression() 
lr.fit(x_train, y_train) 
print("The accuracy score is:", lr.score(x_test, y_test))






#5. Naive Bayesian Classifier

from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 

iris = load_iris() 
x = iris.data 
y = iris.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1) 

classifier = GaussianNB() 
classifier.fit(x_train, y_train) 
print("The accuracy is:", classifier.score(x_test, y_test))







#6. K means Clustering

import numpy as nm 
import matplotlib.pyplot as mtp 
import pandas as pd 
from sklearn.datasets import load_iris 

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" 
new_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'iris_class'] 
dataset = pd.read_csv(url, names=new_names, skiprows=0, delimiter=',') 
print(dataset.head(), "\n") 

x = dataset.iloc[:, 0:-1].values 
from sklearn.cluster import KMeans 

wcss_list = [] 
for i in range(1, 11): 
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42) 
    kmeans.fit(x) 
    wcss_list.append(kmeans.inertia_) 

mtp.plot(range(1, 11), wcss_list) 
mtp.title('The Elbow Method Graph') 

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42) 
y_predict = kmeans.fit_predict(x) 

mtp.xlabel('Number of clusters (k)') 
mtp.ylabel('WCSS') 
mtp.show() 

mtp.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s=100, c='blue', label='Iris-setosa') 
mtp.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s=100, c='green', label='Iris-versicolour') 
mtp.scatter(x[y_predict == 2, 0], x[y_predict == 2, 1], s=100, c='red', label='Iris-virginica') 
mtp.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroid') 

mtp.legend() 
mtp.show()
