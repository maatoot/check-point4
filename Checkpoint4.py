# logistic reg code example

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('titanic-passengers.csv', sep=';')
#values that  can't impact the passenger chance to survive ? 
# name , cabin , 
# cabin has only 204 vals
drops = ['Ticket', 'Cabin','Embarked']
df = df.drop(drops, axis=1)
#-----------------------------------#
# age has a lot of na , we need to convert sex , survived into num values 
#___________________________________#





df['Age'] = df['Age'].fillna(np.mean(df['Age']))
df['Sex'].replace(['female','male'],[0,1],inplace=True)
df['Survived'].replace(['Yes','No'],[0,1],inplace=True)


data = df
df.head()



x= data[["Sex","Age"]]
y= data["Survived"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)  #splitting data
# remember that u have to split the data into binary cat
logreg = LogisticRegression()   #build our logistic model
logreg.fit(x_train, y_train)  #fitting training data
y_pred  = logreg.predict(x_test)    #testing model’s performance
print("Accuracy={:.2f}".format(logreg.score(x_test, y_test)))

sns.regplot(x='Sex',y='Survived',data=data)


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
