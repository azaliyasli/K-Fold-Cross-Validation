import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

# K-Fold Cross Validation
accuracies = cross_val_score(classifier, X_train, y_train, cv=10)
print("Accuracy Mean: ", accuracies.mean())
print("Accuracy Standard Deviation : ", accuracies.std())

acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)