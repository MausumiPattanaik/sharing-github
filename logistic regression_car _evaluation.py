
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#--------------------------------------------------------


# upload the data set
df = pd.read_csv('car_evaluation.csv')


#SUMMARY OF DATA SET
#----------------------------

df.head ()
df.info()
df.describe()


#MISSING VALUE ANALYSIS:
#-------------------------------
df.isna().sum()


#UNIQUE VALUE COUNTS OF DATA SET
#-----------------------------

for i in df.columns:
    print(df[i].value_counts())
    print()



#DATA CLEANING : ( 5more = 5   more =5 )
#----------------------------------

df.doors = df.doors.replace({"5more": 5}).astype('int64')
df.persons = df.persons.replace({"more": 5}).astype('int64')


'''
There are 7 variables in the dataset. All the variables are of categorical data type.

ENCODE THE VALUES:
    vhigh = 4  high=3  med=2  low=1
    small =1   med=2   big=3

'''

#LABEL  ENCODING:
#--------------------------------

df2 = df.copy(True)

df2 .buying  = df2.buying.map({ 'vhigh':4,'high':3, 'med':2,'low':1 })
df2 .maint  = df2.maint.map({ 'vhigh':4,'high':3, 'med':2,'low':1 })
df2.lug_boot = df2.lug_boot.map({ 'big':3, 'med':2,'small':1 })
df2 .safety = df2 . safety.map({ 'high':3, 'med':2,'low':1 })

for i in df2.columns:
    print(df2[i].value_counts())
    print()


# UNIVARIATE ANALYSIS:

sns.countplot(x = 'Evaluation', data = df2, palette = 'hls')
plt.show()

'''
acc - 22%
good - 3.9%
unacc - 70%
vgood - 3.7%
If the model perdicts all input as unacc then it will be 70% accurate which is wrong.
Hence we cannot conclude a model's performance just by accuracy.'''


#CHI_SCORE TEST ( TO CREATE A CORRELATION):
#---------------------------------------------
 
df2_1= df2.replace(['unacc','acc','vgood','good'],[1,2,3,4])

from sklearn.feature_selection import chi2

y = df2_1.Evaluation 
x = df2_1.drop('Evaluation', axis = 1)
chi_scores = chi2(x,y)
chi_scores

Chi2 = pd.Series(chi_scores[0],index= x.columns)
p_values = pd.Series(chi_scores[1],index = x.columns)
p_values.sort_values(ascending = False , inplace = True)

p_values.plot.bar()

corr= df2_1.corr()
sns.heatmap(corr, annot = True,fmt='.1g',cmap= 'coolwarm', linewidths=3, linecolor='black')



'''       modeling:       '''

# ASSIGNED RESPONCE AND EXPLANATOR VARIABLES 
#--------------------------------------------- 

y = df2.Evaluation 
x = df2.drop('Evaluation', axis = 1)



# CROSS VALIDATION (SPLIT DATA INTO TRAINING AND TEST SET )
#---------------------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0) 

x_train.shape , x_test.shape


# SCALING:  STANDADIZE DATASET
#--------------------------------

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train_scale = sc.fit_transform(x_train)
x_test_scale = sc.transform(x_test)


#IMPORT MODEL AND CREATE THE INSTANT:
#----------------------------------
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# TRAIN AND PREDICT THE MODEL:
#-------------------------------
model.fit(x_train_scale,y_train)
y_pred = model.predict(x_test_scale)

model.intercept_
model.coef_


#ACCURACY CALCULATION (TEST THE SCOTE TO CHEK THE MISMATCH) :
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test,y_pred)   

from sklearn.metrics import precision_score, recall_score
precision_score(y_test,y_pred,average='macro')      
recall_score(y_test,y_pred,average= 'macro')   

#DRAW THE CONFUSION MATRIX:
confusion_matrix(y_test,y_pred)

'''
accuracyvalue= 84% (Out[93]: 0.8402777777777778)
array([[ 63,   3,  32,   1],
       [ 11,   9,   0,   1],
       [ 16,   1, 279,   0],
       [  4,   0,   0,  12]], dtype=int64)

'''

from sklearn.metrics import classification_report
print (classification_report(y_test,y_pred))


#k-fold cross validation
from sklearn.model_selection import cross_val_score
cross_val_score(model,x,y,cv=4).mean()







#--------------------------------------------------





