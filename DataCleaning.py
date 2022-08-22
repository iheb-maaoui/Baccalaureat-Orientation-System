import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

def catToNum(df,variable):
    classes = [1,0,2]
    RealClasses = list(df[variable].unique())
    RealClasses.sort()
    for i in range(3):
        df[variable] = np.where(df[variable]==RealClasses[i],classes[i],df[variable])

def catToNum2(df,variable,l):
    ## Pas du tout -----> 0
    ## Moyennement ------> 1
    ## Oui Parfaitement ou Beaucoup------> 2
    classes = [0,1,2]
    RealClasses = ['Pas du tout', 'Moyennement', l]
    for i in range(3):
        df[variable] = np.where(df[variable]==RealClasses[i],classes[i],df[variable])

df = pd.read_csv("P2MUpdated.csv")
pd.set_option('display.max_columns', None)
del df['Horodateur']

labels = list(df['Quel est ton parcours académique ?'].unique())
df['Quel est ton parcours académique ?'] = df['Quel est ton parcours académique ?'].apply(lambda x: labels.index(x))

## 'Combiens de livres as tu  lu les 6 derniers mois?' 
## <2 ----> 0
## 2 à 4 -----> 1
## >4 -----> 2
variable = 'Combiens de livres as tu  lu les 6 derniers mois?'
catToNum(df,variable)

##'Combiens de pages as tu écrites?'
##<5 ----> 0
##5 à 10 -----> 1
##>10 ----->2
catToNum(df,'Combiens de pages as tu écrites?')

## 'As tu fait des tentatives de poésie?'
## Non ----> 0
## 1 à 2 -----> 1
## >2 -----> 2
variable = 'As tu fait des tentatives de poésie?'
df[variable] = np.where(df[variable]=='Non','<2',df[variable])
catToNum(df,'As tu fait des tentatives de poésie?')

## 'Lis tu des livres d'histoire?'
## Non ----> 0
## 1 à 3 ----> 1
## >3 ----> 2
variable = "Lis tu  des livres d'histoire?"
df[variable] = np.where(df[variable]=='Non','<1',df[variable])
catToNum(df,"Lis tu  des livres d'histoire?")

L2 = []
L1 = []
RealClasses = ['Pas du tout', 'Moyennement', 'Oui parfaitement']
for i in df.columns:
    aux = df[i].unique()
    if 'Pas du tout' in aux:
        
            if 'Beaucoup' in aux:
                L1.append(i)
            else:
                L2.append(i)

for variable in L1:
    catToNum2(df,variable,'Beaucoup')

for variable in L2:
    catToNum2(df,variable,'Oui parfaitement')

## 'Combiens de journaux par semaine tu lis?'
## <3 ----> 0
## 3 à 7 -----> 1
## >7 ------->2
variable = 'Combiens de journaux par semaine tu lis?'
catToNum(df,variable)

##Dans le présent gouvernement: combien de ministres peux tu identifier?
## <5 ----> 0
## >5 ----> 1
## Tous ----> 2
variable = "Dans le présent gouvernement: combien de ministres peux tu identifier?"
df[variable] = np.where(df[variable]=='Tous',2,df[variable])
df[variable] = np.where(df[variable]=='<5',0,df[variable])
df[variable] = np.where(df[variable]=='>5',1,df[variable])

## 'Votre moyenne des sciences naturelles en 3eme annee secondaire'
## <10 ----> 0
## 10 à 15 ----> 1
## >15 ----> 2
variable ='Votre moyenne des sciences naturelles en 3eme annee secondaire'
df[variable] = np.where(df[variable]=='>15',2,df[variable])
df[variable] = np.where(df[variable]=='<10',0,df[variable])
df[variable] = np.where(df[variable]=='10 à 15',1,df[variable])

##'Est-ce tu veux te consacrer aux autres?'
## Pas de tout ----> 0
## Un Peu ------> 1 
## Oui souvent ----> 2
variable = 'Est-ce tu veux te consacrer aux autres?'
df[variable] = np.where(df[variable]=='Un peu',1,df[variable])
df[variable] = np.where(df[variable]=='Oui souvent',2,df[variable])



##Handling Missing Values

#Deleting feature with a lot of missing values
del df['Est-ce tu as choisi les math comme option?']
## Most fréquent values technique
for i in df.columns:
    df[i] = np.where(df[i].isnull(),df[i].value_counts().index[[df[i].value_counts().argmax()]][0],df[i])


## One Hot encoding for nominal variables
import category_encoders as ce
#Create object for one-hot encoding
L= ['As tu  visité un monument historique?',"Comment exprimes tu  tes idées et tes émotions?",'Joues tu le scrable?','Votre Réaction face à une nouvelle?',"Est-ce que  tu joues à l'echec?",'Devant une décision:',"Lors d'un discours:","Lors d'une discussion?",'Dans ton travail, tu préfères:']#'Est-ce tu as choisi les math comme option?']  
encoder=ce.OneHotEncoder(cols=L,handle_unknown='return_nan',return_df=True,use_cat_names=True)
data_encoded = encoder.fit_transform(df)

#Correlation Analysis

correlation_matrix=data_encoded.corr()
correlated_features=[]
for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) >= 0.7:
            colname = correlation_matrix.columns[i]
            correlated_features.append(colname)

data_encoded.drop(correlated_features, axis='columns', inplace=True)

#DataFrame to array
X = data_encoded.iloc[:,1:].values
y = data_encoded.iloc[:,0].values
X=X.astype('int')
y=y.astype('int')
#Dataset balancing (Random oversampling method)
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_resample(X, y)
newdf = pd.DataFrame(columns=['Quel est ton parcours académique ?'],data=y[:])

##Features Selection Function (chi-squared)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
def select_features(Xtrain, ytrain, Xtest,i):
    fs = SelectKBest(score_func=chi2, k=i)
    fs.fit(Xtrain, ytrain)
    Xtrain_fs = fs.transform(Xtrain)
    Xtest_fs = fs.transform(Xtest)
    return Xtrain_fs, Xtest_fs, fs
print(y.shape)

np.save('X_cleaned',X)
np.save('y_cleaned',y)

