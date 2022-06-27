#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


import warnings
warnings.filterwarnings('ignore')

import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import joblib
from scipy.stats import boxcox
import seaborn as sns


# import snowflake.connector
import lightgbm as lgb
from tqdm import tqdm
import scipy.stats as stats
from scipy.stats import shapiro
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.metrics import recall_score, accuracy_score, precision_score


# In[3]:


import pandas as pd
df = pd.read_csv ('C:\\Users\\Daniela Neva\\Documents\\NevarYLLover\\Maestría Geomática\\Datos\\Proceso 1 - Modelo\\BD_Tesis\\Datos_iniciales_V0.15.csv', sep=';', encoding='latin-1')
df = pd.DataFrame(df)
df

#df tiene toda la información.


# In[4]:


#Seleccionar datos de geoquímica
GQ=df.iloc[::,7:53:]
GQ = GQ.astype(float)
GQ


# In[24]:


#Seleccionar la variable - elemento.
#Ag_ppm
Ag=df.iloc[:,[5,7]]


#AgProductiva
#Ag_P=Ag.loc[Ag["Productividad"].isin(["Productiva"])]
#AgP_Val=Ag_P.iloc[:,1:2]
#AgP=np.array(AgP_Val)
#print(AgP)
#print(AgP.shape)
#77 x 1

#Seleccionar los datos de Ag de muestras no productivas
Ag_NP=Ag.loc[Ag["Productividad"].isin(["No productiva"])]
#Seleccionar solo los valores numéricos de Ag de muestras no productivas
AgNP_Val=Ag_NP.iloc[:,1:2]
# Ordenarlos como array
AgNP=np.array(AgNP_Val)
#print(AgNP)
print(AgNP.shape)
# 88 x 1

#Test de Shapiro - Probar normalidad para el Ag en muestras productivas y no productivas
print(stats.shapiro(AgNP))
print(stats.shapiro(AgP))

sns.distplot(AgP, hist=False, kde=True)
sns.distplot(AgNP, hist=False, kde=True)

#Since the p-value is less than .05, we reject the null hypothesis. We have sufficient evidence to say that
#the sample data does not come from a normal distribution.


# In[6]:


#No hay normalidad, entonces se aplica boxcox
#Flatten - Poner los datos en una sola dimensión
AgP=(AgP.flatten("A"))
AgNP=(AgNP.flatten("A"))
#Usar boxcox para normalizar los datos
AgP_Norm,_=stats.boxcox(AgP)
AgNP_Norm,_=stats.boxcox(AgNP)
#print(AgP_Norm)
#print(AgNP_Norm)

#Graficar los datos después de boxcox
sns.distplot(AgP_Norm, hist=False, kde=True)
sns.distplot(AgNP_Norm, hist=False, kde=True)


# In[7]:


#Test de Shapiro - Probar normalidad después de BoxCox

print(stats.shapiro(AgP_Norm))
print(stats.shapiro(AgNP_Norm))


# In[8]:


#No hay normalidad en los datos - Aplicar prueba de Wilcoxon sobre los datos originales
#Prueba de dos colas porque estamos comparando las medias de dos grupos 
from scipy.stats import mannwhitneyu
stats.mannwhitneyu(AgP,AgNP,alternative="two-sided")

#pvalue menor a 0.05 - No hay diferencia significativa.


# In[9]:


#Seleccionar datos de concentración de elementos

#Z-Score para normalizar los datos (z = (X – μ) / σ ) #X = datos #μ =Promedio #σ=Desviación estándar
#Normalizar datos de geoquímica
import scipy.stats as stats
GQNorm=stats.zscore(GQ)
GQNorm

sns.distplot(GQNorm, hist=False, kde=True)

#Transpuesta de la matriz porque Numpy trata cada fila de la matriz como una variable separada
GQt=np.transpose(GQNorm)
GQt

#Matriz de correlación - 46 elementos quimicos
mcr=np.corrcoef(GQt)
print(mcr)


# In[47]:


#Graficar la matriz de correlación
import numpy as np
import matplotlib.pyplot as plt



x_axis_labels = GQ.columns.values
y_axis_labels = GQ.columns.values

plt.figure(figsize=(40, 40))
plt.title('Matriz de correlación');
sns.heatmap(mcr, annot=True, linewidth=.1, vmax=1, fmt='.1f', cmap='YlOrRd',center=0,xticklabels=x_axis_labels, yticklabels= y_axis_labels)


# In[49]:


#Zona_Mineral y elementos químicos
data=df.iloc[::,5:53:]


#Variables explicativas
explicativas=data.drop(columns=["Productividad","Método"])
#La variable que quiero predecir es Zona Mineral - Es una variable categórica - Problema de clasificación.
objetivo=data.Productividad


# In[50]:


# Tratamiento de datos
# ------------------------------------------------------------------------------
import statsmodels.api as sm

# Gráficos
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Preprocesado y modelado
# ------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Configuración warnings
# ------------------------------------------------------------------------------
import warnings
warnings.filterwarnings('once')


# In[25]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(explicativas, objetivo, test_size=0.3, random_state=1) # 70% training and 30% test


#Model
Mi_primer_modelo=DecisionTreeClassifier(max_depth=4)
#Train
Mi_primer_modelo.fit(X_train,y_train)
#Predict
y_pred = Mi_primer_modelo.predict(X_test)

#Graphic
plt.figure(figsize=(22,8))
plot_tree(decision_tree=Mi_primer_modelo,feature_names=explicativas.columns, filled=True, fontsize=9);


# In[52]:


#Precisión del modelo

#Train Vs Test
from sklearn import datasets, metrics
from sklearn.metrics import classification_report


A=classification_report(y_test,y_pred)
print(A)


# In[53]:


#Predict whole data
y_pred = Mi_primer_modelo.predict(explicativas)

#Graphic
plt.figure(figsize=(22,8))
plot_tree(decision_tree=Mi_primer_modelo,feature_names=explicativas.columns, filled=True, fontsize=9);


# In[54]:


#Precisión del modelo

#Train Vs Test
from sklearn import datasets, metrics
from sklearn.metrics import classification_report


A=classification_report(objetivo,y_pred)
print(A)


# In[55]:


from sklearn.model_selection import KFold
n_splits = 5
kf = KFold(n_splits=5, shuffle=False)

#Utilizando Cross_Validation
from sklearn.model_selection import cross_val_score
CrossV=cross_val_score(Mi_primer_modelo,explicativas,objetivo,cv=5)
print("Validación cruzada:", CrossV)
pd.DataFrame(CrossV)

#from sklearn.model_selection import cross_validate
kfold = KFold(n_splits=5)

from sklearn.model_selection import cross_validate
kf = cross_validate(Mi_primer_modelo,explicativas,objetivo, cv=kfold)
pd.DataFrame(kf)


# In[56]:


import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
MC=confusion_matrix(objetivo,y_pred)
ax = sns.heatmap(MC, annot=True, fmt='g');
ax.set_title('Zonas minerales - Matriz de confusión - Árbol de decisión' );
ax.set_xlabel('ZM - Predicción')
ax.set_ylabel('ZM - Entrenamiento');
ax.xaxis.set_ticklabels(['P', 'NP'])
ax.yaxis.set_ticklabels(['P', 'NP'])

