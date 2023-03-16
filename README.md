##MACHINE LEARNING PROJECT


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics


dataset = pd.read_csv('Admission_Predict.csv')
dataset.head()
dataset.shape
dataset.info()
dataset.describe()
dataset.isnull().sum()
dataset.corr()
dataset.cov()
dataset.columns

dataset = dataset.drop('Serial No.',axis=1)
X = dataset.drop('Chance of Admit ',axis=1)
y = dataset['Chance of Admit ']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
lr =LinearRegression()
lr.fit(X_train,y_train)
svm = SVR()
svm.fit(X_train,y_train)
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
gr = GradientBoostingRegressor()
gr.fit(X_train,y_train)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train,y_train, train_size=0.20, random_state=42)
print('Train/Test Sets Sizes:', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
from sklearn.ensemble import BaggingRegressor
bag_regressor = BaggingRegressor(random_state = 1)
bag_regressor.fit(X_train, Y_train)
Y_preds = bag_regressor.predict(X_test)
print('Training Coefficient of R2: %.3f'%bag_regressor.score(X_train, Y_train))
print('Test Coefficient of R2: %.3f'%bag_regressor.score(X_test, Y_test)

y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gr.predict(X_test)

score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)

print(score1,score2,score3,score4)

final_data = pd.DataFrame({'Models':['LR','SVR','RF','GR'],
 'R2_Score':[score1,score2,score3,score4]})
final_data

# sns.barplot(final_data['Models'],final_data['R2_Score'])
# plt.show()
y_train = [1 if value>0.8 else 0 for value in y_train]
y_test = [1 if value>0.8 else 0 for value in y_test]
y_train = np.array(y_train)
y_test = np.array(y_test)
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred1= lr.predict(X_test)
print(accuracy_score(y_test,y_pred1))
svm = svm.SVC()
svm.fit(X_train,y_train)
y_pred2 = svm.predict(X_test)
print(accuracy_score(y_test,y_pred2))
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred3 = knn.predict(X_test)
print(accuracy_score(y_test,y_pred3))

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred4 = rf.predict(X_test)
print(accuracy_score(y_test,y_pred4))
In [43]:
gr = GradientBoostingClassifier()
gr.fit(X_train,y_train)
y_pred5 = gr.predict(X_test)
print(accuracy_score(y_test,y_pred5))
final_data = pd.DataFrame({'Models':['LR','SVC','KNN','RF','GBC'],
 Models Accuracy_Score
               'Accuracy_Score':[accuracy_score(y_test,y_pred1),
                 accuracy_score(y_test,y_pred2),
                 accuracy_score(y_test,y_pred3),
                  accuracy_score(y_test,y_pred4),
                    accuracy_score(y_test,y_pred5)]})
final_data



from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import xgboost as xgb


estimators == [
               ('rf', RandomForestClassifier(n_estimators = 10, random_state = 42
              ('knn', KNeighborsClassifier(n_neighbors = 10)),
             ('gbc', GradientBoostingClassifier()),
             ('lr', LogisticRegression()),
             ('ccv', CalibratedClassifierCV()),
             ('mlp', MLPClassifier()),
              ('dt', DecisionTreeClassifier()),
              ('lda', LinearDiscriminantAnalysis()),
              ('gnb', GaussianNB()),
              ('adb', AdaBoostClassifier()),
              ('etc', ExtraTreesClassifier()),
              ('sgd', SGDClassifier()),
              ('svm', SVC()),
              ('xgb', xgb.XGBClassifier(n_estimators= 10, random_state = 42)) ]

b.XGBClassifier(n_estimators= 10, random_state = 42)) ]

from sklearn.ensemble import StackingClassifier
clf = StackingClassifier(
 estimators = estimators, final_estimator = LogisticRegression(),cv = 10,
 stack_method='predict',n_jobs= -1
     )

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

accuracy_score(y_test,y_pred)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

DTC_parameters = {
              'criterion' : ['gini', 'entropy', 'log_loss'],
                  'splitter' : ['best', 'random'],
                 'max_depth' : range(1,10,1),
                 'min_samples_split' : range(2,10,2),
                  'min_samples_leaf' : range(1,5,1),
                  'max_features' : ['auto', 'sqrt', 'log2']
}

Bagging_parameters = {
                  'n_estimators' : [5, 10, 15],
                   'max_samples' : range(2, 10, 1),
                   'max_features' : range(2, 10, 3)
}

RFC_parameters = {
                  'criterion' : ['gini', 'entropy', 'log_loss'],
                    'max_depth' : range(1, 10, 1),
                       'min_samples_split' : range(2, 10, 2),
                       'min_samples_leaf' : range(1, 10, 1),
}

ETC_parameters = {
                 'n_estimators' : [10,20,30],
                   'criterion' : ['gini', 'entropy', 'log_loss'],
                    'max_depth' : range(2,10,1),
                     'min_samples_split' : range(2,10,2),
                     'min_samples_leaf' : range(1,5,1),
                      'max_features' : ['sqrt', 'log2']
}

hyper1 = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = DTC_pa
hyper2 = GridSearchCV(estimator = BaggingClassifier(), param_grid = Bagging_par
hyper3 = GridSearchCV(estimator = RandomForestClassifier(), param_grid = RFC_pa
hyper4 = GridSearchCV(estimator = ExtraTreesClassifier(), param_grid = ETC_para

hyper1.fit(X_train,y_train)
hyper2.fit(X_train,y_train)
hyper3.fit(X_train,y_train)
hyper4.fit(X_train,y_train)

h_pred1 = hyper1.predict(X_test)
h_pred2 = hyper1.predict(X_test)
h_pred3 = hyper1.predict(X_test)
h_pred4 = hyper1.predict(X_test)

accuracy_score(y_test,h_pred1)
accuracy_score(y_test,h_pred2)
accuracy_score(y_test,h_pred3)
accuracy_score(y_test,h_pred4)

X = dataset.drop('Chance of Admit ',axis=1)
y = dataset['Chance of Admit ']
y = [1 if value>0.8 else 0 for value in y]
y = np.array(y)
X = sc.fit_transform(X)
gr = GradientBoostingClassifier()
gr.fit(X,y)


#file download
import joblib
joblib.dump(gr,'admission_model')
model = joblib.load('admission_model')
model.predict(sc.transform([[337,118,4,4.5,4.5,9.65,1]]))


from tkinter import *
import joblib
from sklearn.preprocessing import StandardScaler

def show_entry():
             p1 = float(e1.get())
             p2 = float(e2.get())
             p3 = float(e3.get())
             p4 = float(e4.get())
             p5 = float(e5.get())
             p6 = float(e6.get())
             p7 = float(e6.get())
             model = joblib.load('admission_model')
       result = model.predict(sc.transform([[p1,p2,p3,p4,p5,p6,p7]]))
 
    if result == 1:
                Label(master, text='High Chance of Getting Admission').grid(row=31)
    else:
         Label(master, text='You May Get Admission').grid(row=31)
 
master =Tk()
master.title('Graduate Admission Analysis and Prediction')
label = Label(master,text = 'Graduate Admission Analysis & Prediction',bg = "bl
                 fg = "white").grid(row=0,columnspan=2)
Label(master,text = 'Enter Your GRE Score').grid(row=1)
Label(master,text = 'Enter Your TOEFL Score').grid(row=2)
Label(master,text = 'Enter University Rating').grid(row=3)
Label(master,text = 'Enter SOP').grid(row=4)
Label(master,text = 'Enter LOR').grid(row=5)
Label(master,text = 'Enter Your CPGA').grid(row=6)
Label(master,text = 'Research').grid(row=7)
e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)
e7.grid(row=7,column=1)
Button(master,text='Predict',command=show_entry).grid()
mainloop()
