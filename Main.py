"""
Created on Mon Oct 16 12:27:48 2023

@author: sunhouse

After checking the possibility of modeling ( in try.py) we cna go for main Code.
The second edition

Nokte asli:
    
    ma omadim az gridsearch estefade krdim va too mape yek model bhtr shod
    ama vaghty omdim haamro ba hamon hyperparameetr zadim motasefane
    modelaye dg behtr amal krdn
    
    ghazie injas k ma bayad ba range kar konim
    
    nokte injas vase hamin nested cross val estefade mishe
    
"""


#============================================================================
'                                 IMPORT                                '
#============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

from sklearn.compose import TransformedTargetRegressor
from  sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor



import seaborn as sns


#============================================================================
'                                 LOAD DATA                                '
#============================================================================


x=np.array(([100,0],[80,20],[50,50],[20,80],[0,100]))
y1=np.array([66,65,64,61,61])
y2=np.array((52,54,55,51,40))




#============================================================================
'                                CORRELATION                                '
#============================================================================


datacor=pd.DataFrame(x)
datacorr=pd.concat([datacor,pd.DataFrame(y1),pd.DataFrame(y2)],axis=1)
plt.figure(figsize=(10, 6))  # Set the figure size (width: 10 inches, height: 6 inches)
plt.rcParams["figure.dpi"] = 600 
plt.rcParams['font.family']='Arial'

correlation = datacorr.corr()  
xt=['PEO %','PCL %','Tm','Tc']
yt=['PEO %','PCL %','Tm','Tc']
sns.heatmap(correlation,cmap="coolwarm",xticklabels=xt,
            yticklabels=yt)
plt.tick_params(labelsize=20,pad=12)

name='correlation.jpg'
plt.savefig(name,dpi=600,format='jpg')



#============================================================================
'                                HYPERPARAMETER                                '
#============================================================================


def select_model(x,y,model,score='MAPE'):
    if score=='MAPE':
        sc='neg_mean_absolute_percentage_error'
    if score=='MAE':
        sc='neg_mean_absolute_error'
    if score=='MSE':
        sc='neg_mean_squared_error'
    if score=='RMSE':
        sc='neg_root_mean_squared_error'
        
    
    
    
    if model=='LR':
        scaler=MinMaxScaler()
        poly=PolynomialFeatures()
        regressior=LinearRegression()        
        params={'scaler':[None,MinMaxScaler(),StandardScaler(),RobustScaler()],
                'poly__degree':[1,2,3]}
        pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
    if model== 'KNN':
        scaler=MinMaxScaler()
        regressior = KNeighborsRegressor()
        params={'scaler':[None,MinMaxScaler(),StandardScaler(),RobustScaler()],
                'regressior__n_neighbors':[1,2,3]}
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)]) 
    if model== 'DT':
        scaler=MinMaxScaler()
        regressior=DecisionTreeRegressor(random_state=0)
        params={'scaler':[None,MinMaxScaler(),StandardScaler(),RobustScaler()],
                'regressior__max_depth':[1,2,3,4,5],
         'regressior__min_samples_split':[2,3]}     
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
    if model=='RF':
        scaler=MinMaxScaler()
        regressior=RandomForestRegressor(random_state=0)
        params={'scaler':[None,MinMaxScaler(),StandardScaler(),RobustScaler()],
                'regressior__n_estimators':[5,10,20],
                    'regressior__max_depth':[1,2,3,4,5],
                    'regressior__min_samples_split':[2,3]}
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
    if model=='SVR':
        scaler=MinMaxScaler()
        regressior = SVR(max_iter=10000)
        params=[{'scaler':[None,MinMaxScaler(),StandardScaler(),RobustScaler()],
                 'regressior__kernel': ['rbf'],
                    'regressior__gamma':[0.0001,0.1,1,100],
                    'regressior__C':[0.0001,0.1,1,100]},
                
                    {'scaler':[None,MinMaxScaler(),StandardScaler(),RobustScaler()],
                     'regressior__kernel':['linear'],
                    'regressior__C':[0.0001,0.1,1,10]},
                    
                    
                    {'scaler':[None,MinMaxScaler(),StandardScaler(),RobustScaler()],
                     'regressior__kernel': ['poly'],
                    'regressior__C':[0.0001,0.1,1,10],
                    'regressior__degree':[1,2,3]}]  

           
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
    if model=='MLP':   
        regressior=MLPRegressor(solver='adam',max_iter=10000,random_state=40)
        scaler=MinMaxScaler()
        params={'scaler':[MinMaxScaler(),StandardScaler()],
                'regressior__hidden_layer_sizes':[(10,10),
                                                  (100,)],
                    'regressior__activation':[ 'tanh', 'relu'],
                    'regressior__alpha':[0.001,0.1]}
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        
        
    kfold1=KFold(n_splits=5,shuffle=True,random_state=0) 
   # scoring_list='neg_mean_absolute_error'
    scoring_list=sc
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list)

    grid.fit(x,y)
    return grid






'''
if which=='prediction':
    best=20
    cross_pred=cross_val_predict(best, x,y,cv=kfold1,
                            n_jobs=-1)
    return cross_pred
'''

#----------for repeating but for MSE,MAE,RMSE , decresase hyperparameter
def select_model1(x,y,model,score='MAPE'):
    if score=='MAPE':
        sc='neg_mean_absolute_percentage_error'
    if score=='MAE':
        sc='neg_mean_absolute_error'
        print('mae')
    if score=='MSE':
        sc='neg_mean_squared_error'
    if score=='RMSE':
        sc='neg_root_mean_squared_error'
        print('rmse')
        
    
    
    
    if model=='LR':
        scaler=MinMaxScaler()
        poly=PolynomialFeatures()
        regressior=LinearRegression()        
        params={'scaler':[None],
                'poly__degree':[1]}
        pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
    if model== 'KNN':
        scaler=MinMaxScaler()
        regressior = KNeighborsRegressor()
        params={'scaler':[None],
                'regressior__n_neighbors':[1]}
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)]) 
    if model== 'DT':
        scaler=MinMaxScaler()
        regressior=DecisionTreeRegressor(random_state=0)
        params={'scaler':[None],
                'regressior__max_depth':[1],
         'regressior__min_samples_split':[2]}     
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
    if model=='RF':
        scaler=MinMaxScaler()
        regressior=RandomForestRegressor(random_state=0)
        params={'scaler':[None],
                'regressior__n_estimators':[5],
                    'regressior__max_depth':[2],
                    'regressior__min_samples_split':[2]}
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
    if model=='SVR':
        scaler=MinMaxScaler()
        regressior = SVR(max_iter=10000)
        params={'scaler':[StandardScaler()],
                     'regressior__kernel':['linear'],
                    'regressior__C':[1]} 

           
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
    if model=='MLP':   
        regressior=MLPRegressor(solver='adam',max_iter=10000,random_state=40)
        scaler=MinMaxScaler()
        params={'scaler':[MinMaxScaler()],
                'regressior__hidden_layer_sizes':[
                                                  (100,)],
                    'regressior__activation':['relu'],
                    'regressior__alpha':[0.1]}
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        
        
    kfold1=KFold(n_splits=5,shuffle=True,random_state=0) 
   # scoring_list='neg_mean_absolute_error'
    scoring_list=sc
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list)

    grid.fit(x,y)
    return grid




def select_model2(x,y,model,score='MAPE'):
    if score=='MAPE':
        sc='neg_mean_absolute_percentage_error'
    if score=='MAE':
        sc='neg_mean_absolute_error'
        print('mae')
    if score=='MSE':
        sc='neg_mean_squared_error'
    if score=='RMSE':
        sc='neg_root_mean_squared_error'
        print('rmse')
    
    
    
    if model=='LR':
        scaler=MinMaxScaler()
        poly=PolynomialFeatures()
        regressior=LinearRegression()        
        params={'scaler':[RobustScaler()],
                'poly__degree':[3]}
        pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
    if model== 'KNN':
        scaler=MinMaxScaler()
        regressior = KNeighborsRegressor()
        params={'scaler':[None],
                'regressior__n_neighbors':[3]}
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)]) 
    if model== 'DT':
        scaler=MinMaxScaler()
        regressior=DecisionTreeRegressor(random_state=0)
        params={'scaler':[None],
                'regressior__max_depth':[1],
         'regressior__min_samples_split':[2]}     
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
    if model=='RF':
        scaler=MinMaxScaler()
        regressior=RandomForestRegressor(random_state=0)
        params={'scaler':[None],
                'regressior__n_estimators':[10],
                    'regressior__max_depth':[1],
                    'regressior__min_samples_split':[3]}
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
    if model=='SVR':
        scaler=MinMaxScaler()
        regressior = SVR(max_iter=10000)
        params={'scaler':[None],
                     'regressior__kernel': ['poly'],
                    'regressior__C':[1],
                    'regressior__degree':[3]} 

           
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
    if model=='MLP':   
        regressior=MLPRegressor(solver='adam',max_iter=10000,random_state=40)
        scaler=MinMaxScaler()
        params={'scaler':[StandardScaler()],
                'regressior__hidden_layer_sizes':[(10,10)],
                    'regressior__activation':[ 'tanh'],
                    'regressior__alpha':[0.1]}
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        
        
    kfold1=KFold(n_splits=5,shuffle=True,random_state=0) 
   # scoring_list='neg_mean_absolute_error'
    scoring_list=sc
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list)

    grid.fit(x,y)
    return grid





#============================================================================
'                             HEATMAP                                    '
#============================================================================


model_list=['LR','KNN','DT','RF','SVR','MLP']


gs=select_model1(x,y1,'LR',score='MAPE')
gs.best_score_
cv=gs.cv_results_

a=float(cv['split0_test_score'])+float(cv['split1_test_score'])+ float(cv['split2_test_score'])+float(cv['split3_test_score'])+float(cv['split4_test_score'])
b=a/5
print(b)
#
#
titsc='RMSE'
sc_list1=[]
#best_list1=[]

for i in range(0,6):
    gs=select_model1(x,y1,model_list[i],score=titsc)
    score=gs.best_score_
    sc_list1.append(score)
    #print(gs.best_params_)
    #best_list1.append(gs.best_params_)
    
    
final1=pd.DataFrame(data=((sc_list1)),
                   index=model_list)

name='Tm_'+titsc+'.csv'
final1.to_csv(name)
print('The file is saved')





titsc='RMSE'

sc_list2=[]
#best_list2=[]
for i in range(0,6):
    gs=select_model2(x,y2,model_list[i],score=titsc)
    score=gs.best_score_
    #print(gs.best_params_)
    #best_list2.append(gs.best_params_)
    sc_list2.append(score)
    
final2=pd.DataFrame(data=((sc_list2)),
                   index=model_list)

name='Tc_'+titsc+'.csv'
final2.to_csv(name)
print('The file is saved')






#---
titsc='MAPE'
sc_list1=[]
#best_list1=[]

for i in range(0,6):
    gs=select_model1(x,y1,model_list[i],score=titsc)
    score=gs.best_score_
    sc_list1.append(score)
    #print(gs.best_params_)
    #best_list1.append(gs.best_params_)
    
    
final1=pd.DataFrame(data=((sc_list1)),
                   index=model_list)

name='Tm_'+titsc+'.csv'
final1.to_csv(name)
print('The file is saved')





titsc='MAPE'

sc_list2=[]
#best_list2=[]
for i in range(0,6):
    gs=select_model2(x,y2,model_list[i],score=titsc)
    score=gs.best_score_
    #print(gs.best_params_)
    #best_list2.append(gs.best_params_)
    sc_list2.append(score)
    
final2=pd.DataFrame(data=((sc_list2)),
                   index=model_list)

name='Tc_'+titsc+'.csv'
final2.to_csv(name)
print('The file is saved')


#----


titsc='MSE'
sc_list1=[]
#best_list1=[]

for i in range(0,6):
    gs=select_model1(x,y1,model_list[i],score=titsc)
    score=gs.best_score_
    sc_list1.append(score)
    #print(gs.best_params_)
    #best_list1.append(gs.best_params_)
    
    
final1=pd.DataFrame(data=((sc_list1)),
                   index=model_list)

name='Tm_'+titsc+'.csv'
final1.to_csv(name)
print('The file is saved')





titsc='MSE'

sc_list2=[]
#best_list2=[]
for i in range(0,6):
    gs=select_model2(x,y2,model_list[i],score=titsc)
    score=gs.best_score_
    #print(gs.best_params_)
    #best_list2.append(gs.best_params_)
    sc_list2.append(score)
    
final2=pd.DataFrame(data=((sc_list2)),
                   index=model_list)

name='Tc_'+titsc+'.csv'
final2.to_csv(name)
print('The file is saved')



#-----
titsc='MAE'
sc_list1=[]
#best_list1=[]

for i in range(0,6):
    gs=select_model1(x,y1,model_list[i],score=titsc)
    score=gs.best_score_
    sc_list1.append(score)
    #print(gs.best_params_)
    #best_list1.append(gs.best_params_)
    
    
final1=pd.DataFrame(data=((sc_list1)),
                   index=model_list)

name='Tm_'+titsc+'.csv'
final1.to_csv(name)
print('The file is saved')





titsc='MAE'

sc_list2=[]
#best_list2=[]
for i in range(0,6):
    gs=select_model2(x,y2,model_list[i],score=titsc)
    score=gs.best_score_
    #print(gs.best_params_)
    #best_list2.append(gs.best_params_)
    sc_list2.append(score)
    
final2=pd.DataFrame(data=((sc_list2)),
                   index=model_list)

name='Tc_'+titsc+'.csv'
final2.to_csv(name)
print('The file is saved')

#======================================================
#======================================================
#----------------------plot---------------------------
#======================================================
#======================================================



file1_location = '//Users//apm//Desktop//Project//Hojjat Emami//pherulite//data//scores//Tm_MAPE' +'.csv'
#f1=open(file1_location,'r')
data1=pd.read_csv(file1_location,index_col=0)
data1=-1*np.array(data1).reshape(1,-1)


file2_location = '//Users//apm//Desktop//Project//Hojjat Emami//pherulite//data//scores//Tm_MAE' +'.csv'
#f1=open(file1_location,'r')
data2=pd.read_csv(file2_location,index_col=0)
data2=-1*np.array(data2).reshape(1,-1)


file3_location = '//Users//apm//Desktop//Project//Hojjat Emami//pherulite//data//scores//Tm_MSE' +'.csv'
#f1=open(file1_location,'r')
data3=pd.read_csv(file3_location,index_col=0)
data3=-1*np.array(data3).reshape(1,-1)


file4_location = '//Users//apm//Desktop//Project//Hojjat Emami//pherulite//data//scores//Tm_RMSE' +'.csv'
#f1=open(file1_location,'r')
data4=pd.read_csv(file4_location,index_col=0)
data4=-1*np.array(data4).reshape(1,-1)

#final1=pd.concat([data1,data2,data3,data4],axis=1)





#final2=pd.concat([data21,data22,data23,data24])





plt.figure(figsize=(20, 12))  # Set the figure size (width: 10 inches, height: 6 inches)
plt.rcParams["figure.dpi"] = 600 
plt.rcParams['font.family']='Arial'
plt.subplot(2,2,1)

yt=['MAPE']
#yt=['MAPE']
xt=['LR','KNN','DT','RF','SVR','MLP']

#np.array(data1).reshape(1,-1)
#yticklabels=yt,xticklabels=xt

ax=sns.heatmap(data1, cmap="cividis", annot=True,
            fmt='.3f',yticklabels=yt,xticklabels=xt,
            annot_kws={"size":14})
ax=plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
ax.tick_params(axis='both',which='both',direction='in',
               bottom=True,top=False,left=True,right=False,labelsize=20,
               pad=12)
cbar=ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

#======================================================================
plt.subplot(2,2,2)

yt=['MAE']
#yt=['MAPE']
xt=['LR','KNN','DT','RF','SVR','MLP']

#np.array(data1).reshape(1,-1)
#yticklabels=yt,xticklabels=xt

ax=sns.heatmap(data2, cmap="cividis", annot=True,
            fmt='.2f',yticklabels=yt,xticklabels=xt,
            annot_kws={"size":17})
ax=plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
ax.tick_params(axis='both',which='both',direction='in',
               bottom=True,top=False,left=True,right=False,labelsize=20,
               pad=12)
cbar=ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

#======================================================================

plt.subplot(2,2,3)

yt=['MSE']
#yt=['MAPE']
xt=['LR','KNN','DT','RF','SVR','MLP']

#np.array(data1).reshape(1,-1)
#yticklabels=yt,xticklabels=xt

ax=sns.heatmap(data3, cmap="cividis", annot=True,
            fmt='.2f',yticklabels=yt,xticklabels=xt,
            annot_kws={"size":17})
ax=plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
ax.tick_params(axis='both',which='both',direction='in',
               bottom=True,top=False,left=True,right=False,labelsize=20,
               pad=12)

cbar=ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

#======================================================================

plt.subplot(2,2,4)

yt=['RMSE']
#yt=['MAPE']
xt=['LR','KNN','DT','RF','SVR','MLP']

#np.array(data1).reshape(1,-1)
#yticklabels=yt,xticklabels=xt

ax=sns.heatmap(data4, cmap="cividis", annot=True,
            fmt='.2f',yticklabels=yt,xticklabels=xt,
            annot_kws={"size":17})

ax=plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
ax.tick_params(axis='both',which='both',direction='in',
               bottom=True,top=False,left=True,right=False,labelsize=20,
               pad=12)
cbar=ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

#======================================================================
name='Tm_score.jpg'

plt.savefig(name,dpi=600,format='jpg')

name='Tm_score.pdf'

plt.savefig(name,dpi=600,format='pdf')



#===================================================
#=====================================================
#======================================================

file21_location = '//Users//apm//Desktop//Project//Hojjat Emami//pherulite//data//scores//Tc_MAPE' +'.csv'
#f1=open(file1_location,'r')
data21=pd.read_csv(file21_location,index_col=0)
data21=-1*np.array(data21).reshape(1,-1)

file22_location = '//Users//apm//Desktop//Project//Hojjat Emami//pherulite//data//scores//Tc_MAE' +'.csv'
#f1=open(file1_location,'r')
data22=pd.read_csv(file22_location,index_col=0)
data22=-1*np.array(data22).reshape(1,-1)

file23_location = '//Users//apm//Desktop//Project//Hojjat Emami//pherulite//data//scores//Tc_MSE' +'.csv'
#f1=open(file1_location,'r')
data23=pd.read_csv(file23_location,index_col=0)
data23=-1*np.array(data23).reshape(1,-1)

file24_location = '//Users//apm//Desktop//Project//Hojjat Emami//pherulite//data//scores//Tc_RMSE' +'.csv'
#f1=open(file1_location,'r')
data24=pd.read_csv(file24_location,index_col=0)
data24=-1*np.array(data24).reshape(1,-1)


plt.figure(figsize=(20, 12))  # Set the figure size (width: 10 inches, height: 6 inches)
plt.rcParams["figure.dpi"] = 600 
plt.rcParams['font.family']='Arial'
plt.subplot(2,2,1)

yt=['MAPE']
#yt=['MAPE']
xt=['LR','KNN','DT','RF','SVR','MLP']

#np.array(data1).reshape(1,-1)
#yticklabels=yt,xticklabels=xt

ax=sns.heatmap(data21, cmap="inferno", annot=True,
            fmt='.3f',yticklabels=yt,xticklabels=xt,
            annot_kws={"size":14})
ax=plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
ax.tick_params(axis='both',which='both',direction='in',
               bottom=True,top=False,left=True,right=False,labelsize=20,
               pad=12)
cbar=ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

#======================================================================
plt.subplot(2,2,2)

yt=['MAE']
#yt=['MAPE']
xt=['LR','KNN','DT','RF','SVR','MLP']

#np.array(data1).reshape(1,-1)
#yticklabels=yt,xticklabels=xt

ax=sns.heatmap(data22, cmap="inferno", annot=True,
            fmt='.2f',yticklabels=yt,xticklabels=xt,
            annot_kws={"size":17})
ax=plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
ax.tick_params(axis='both',which='both',direction='in',
               bottom=True,top=False,left=True,right=False,labelsize=20,
               pad=12)
cbar=ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

#======================================================================

plt.subplot(2,2,3)

yt=['MSE']
#yt=['MAPE']
xt=['LR','KNN','DT','RF','SVR','MLP']

#np.array(data1).reshape(1,-1)
#yticklabels=yt,xticklabels=xt

ax=sns.heatmap(data23, cmap="inferno", annot=True,
            fmt='.2f',yticklabels=yt,xticklabels=xt,
            annot_kws={"size":17})
ax=plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
ax.tick_params(axis='both',which='both',direction='in',
               bottom=True,top=False,left=True,right=False,labelsize=20,
               pad=12)

cbar=ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

#======================================================================

plt.subplot(2,2,4)

yt=['RMSE']
#yt=['MAPE']
xt=['LR','KNN','DT','RF','SVR','MLP']

#np.array(data1).reshape(1,-1)
#yticklabels=yt,xticklabels=xt

ax=sns.heatmap(data24, cmap="inferno", annot=True,
            fmt='.2f',yticklabels=yt,xticklabels=xt,
            annot_kws={"size":17})

ax=plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
ax.tick_params(axis='both',which='both',direction='in',
               bottom=True,top=False,left=True,right=False,labelsize=20,
               pad=12)
cbar=ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

#======================================================================



name='Tc_score.pdf'

plt.savefig(name,dpi=600,format='pdf')



name='Tc_score.jpg'

plt.savefig(name,dpi=600,format='jpg')



#============================================================================
'                             PREDICTION                                    '
#============================================================================
import matplotlib.colors as mcolors
mcolors.CSS4_COLORS



xyfont={'family':'Arial','size':20,'fontweight':'bold'}
xyyfont={'family':'Arial','size':50,'fontweight':'bold'}
tfont={'family':'Arial','size':20}
lfont={'family':'Arial','size':20}


n2=np.arange(0,101).reshape(-1,1)
n1=np.flip(n2).reshape(-1,1)
new=np.concatenate([n1,n2],axis=1)


gs=select_model(x, y1,'LR', score='MAPE')
y_pred1=gs.predict(new)
gs=select_model(x, y1,'KNN', score='MAPE')
y_pred2=gs.predict(new)
gs=select_model(x, y1,'DT', score='MAPE')
y_pred3=gs.predict(new)
gs=select_model(x, y1,'RF', score='MAPE')
y_pred4=gs.predict(new)
gs=select_model(x, y1,'SVR', score='MAPE')
y_pred5=gs.predict(new)
gs=select_model(x, y1,'MLP', score='MAPE')
y_pred6=gs.predict(new)


#For Tm
coll=['#000000','#FFFF00','#4169E1','#00FF00','#9932CC','#FF1493']
models=['LR','KNN','DT','RF','SVR','MLP']

yy=[y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6]
alph=['a','b','c','d','e','f']

for i in range(0,6):
    plt.figure(figsize=(12, 8))  # Set the figure size (width: 10 inches, height: 6 inches)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['font.family']='Arial'
    #plt.subplot(3,2,i+1)
    
    #gray=#000080  
    plt.scatter(n2,yy[i],label=models[i]+' Predicted',c=coll[i],marker='^',alpha=0.7,s=60)
    #c='#808080',marker='s'
    #c='#FF1493',marker='d'
    #'#000080'
    #plt.scatter(x_count,y_mo[:,2],label='ST multi outputs')
    x_countt=np.array((0,20,50,80,100)).reshape(5,1)
    #marker=7
    plt.scatter(x[:,1],y1.reshape(-1,1),c='red',label='Experimental',marker='*',s=180)
    plt.title(alph[i],fontdict=xyyfont)
    plt.xlabel('PCL Concentration (wt%)',fontdict=xyfont,labelpad=20) # X-Label
    plt.ylabel('Tm',fontdict=xyfont,labelpad=20) # Y-Label
    plt.legend(loc='upper center',bbox_to_anchor=(0.83,1.01),fontsize=20)
    ax=plt.gca()
    ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False
                   ,labelsize=20,pad=12)
    plt.grid(alpha=0.5,zorder=1)
    name='TM_'+str(i)+'.jpg'
    plt.savefig(name,dpi=300,format='jpg')





#-----better design
import matplotlib.colors as mcolors
mcolors.CSS4_COLORS



xyfont={'family':'Arial','size':20,'fontweight':'bold'}
xyyfont={'family':'Arial','size':50,'fontweight':'bold'}
tfont={'family':'Arial','size':20}
lfont={'family':'Arial','size':20}


n2=np.arange(0,101).reshape(-1,1)
n1=np.flip(n2).reshape(-1,1)
new=np.concatenate([n1,n2],axis=1)


gs=select_model(x, y1,'LR', score='MAPE')
y_pred1=gs.predict(new)
gs=select_model(x, y1,'KNN', score='MAPE')
y_pred2=gs.predict(new)
gs=select_model(x, y1,'DT', score='MAPE')
y_pred3=gs.predict(new)
gs=select_model(x, y1,'RF', score='MAPE')
y_pred4=gs.predict(new)
gs=select_model(x, y1,'SVR', score='MAPE')
y_pred5=gs.predict(new)
gs=select_model(x, y1,'MLP', score='MAPE')
y_pred6=gs.predict(new)

import seaborn as sns

# Example data (replace this with your actual data)
data1 = np.random.rand(1, 6) * 100

sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 8))

coll = sns.color_palette("husl", 6)  # Seaborn color palette
models = ['LR', 'KNN', 'DT', 'RF', 'SVR', 'MLP']

yy = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6]
alph = ['a', 'b', 'c', 'd', 'e', 'f']

for i in range(0, 6):
    plt.scatter(n2, yy[i], label=models[i] + ' Predicted', c=coll[i], marker='o', alpha=0.7, s=80)
    plt.scatter(x[:, 1], y1.reshape(-1, 1), c='red', label='Experimental', marker='*', s=180)
    plt.xlabel('PCL Concentration (wt%)', fontweight='bold', fontsize=16, labelpad=20)
    plt.ylabel('Tm', fontweight='bold', fontsize=16, labelpad=20)

    plt.legend(loc='upper center', bbox_to_anchor=(0.83, 1.01), fontsize=14)
    sns.despine()

    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=False, left=True, right=False,
                   labelsize=14, pad=12)

    plt.grid(alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()
    

#===best design
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example data (replace this with your actual data)
data1 = np.random.rand(1, 6) * 100

sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(18, 12))

coll = sns.color_palette("husl", 6)  # Seaborn color palette
models = ['LR', 'KNN', 'DT', 'RF', 'SVR', 'MLP']

yy = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6]
alph = ['a', 'b', 'c', 'd', 'e', 'f']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, ax in enumerate(axes.flatten()):
    ax.scatter(n2, yy[i], label=models[i] + ' Predicted', c=coll[i], marker='o', alpha=0.7, s=80)
    ax.scatter(x[:, 1], y1.reshape(-1, 1), c='red', label='Experimental', marker='*', s=180)

    ax.set_title(alph[i], fontweight='bold', fontsize=40,loc='left')
    ax.set_xlabel('PCL Concentration (wt%)', fontweight='bold', fontsize=24, labelpad=20)
    ax.set_ylabel('Tm', fontweight='bold', fontsize=24, labelpad=20)

    ax.legend(loc='best', fontsize=20)

    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=False, left=True, right=False,
                   labelsize=14, pad=12)

    ax.grid(alpha=0.5, linestyle='--')

plt.tight_layout()
#plt.show()

name='TM_predicted.jpg'
plt.savefig(name,dpi=300,format='jpg')


name='TM_predicted.pdf'
plt.savefig(name,dpi=300,format='pdf')


name='TM_predicted.jpg'
plt.savefig(name,dpi=300,format='jpg')







#===========ttc==========================

    
gs=select_model(x, y2,'LR', score='MAPE')
y_pred21=gs.predict(new)
gs=select_model(x, y2,'KNN', score='MAPE')
y_pred22=gs.predict(new)
gs=select_model(x, y2,'DT', score='MAPE')
y_pred23=gs.predict(new)
gs=select_model(x, y2,'RF', score='MAPE')
y_pred24=gs.predict(new)
gs=select_model(x, y2,'SVR', score='MAPE')
y_pred25=gs.predict(new)
gs=select_model(x, y2,'MLP', score='MAPE')
y_pred26=gs.predict(new)



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example data (replace this with your actual data)
data1 = np.random.rand(1, 6) * 100

sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(18, 12))

coll = sns.color_palette("husl", 6)  # Seaborn color palette
models = ['LR', 'KNN', 'DT', 'RF', 'SVR', 'MLP']

yy=[y_pred21,y_pred22,y_pred23,y_pred24,y_pred25,y_pred26]
alph = ['a', 'b', 'c', 'd', 'e', 'f']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, ax in enumerate(axes.flatten()):
    ax.scatter(n2, yy[i], label=models[i] + ' Predicted', c=coll[i], marker='o', alpha=0.7, s=80)
    ax.scatter(x[:, 1], y2.reshape(-1, 1), c='red', label='Experimental', marker='*', s=180)

    ax.set_title(alph[i], fontweight='bold', fontsize=40,loc='left')
    ax.set_xlabel('PCL Concentration (wt%)', fontweight='bold', fontsize=24, labelpad=20)
    ax.set_ylabel('Tc', fontweight='bold', fontsize=24, labelpad=20)

    ax.legend(loc='best', fontsize=20)

    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=False, left=True, right=False,
                   labelsize=14, pad=12)

    ax.grid(alpha=0.5, linestyle='--')

plt.tight_layout()
#plt.show()
name='TC_predicted.pdf'
plt.savefig(name,dpi=300,format='pdf')


name='TC_predicted.jpg'
plt.savefig(name,dpi=300,format='jpg')



    

    
    
    
    
    
    
    
    
    

#============================================================================
'                   hypothesis minitab                         '
#============================================================================

model_list=['LR','KNN','DT','RF','SVR','MLP']
titsc='MAPE'
sc_list1=[]
#best_list1=[]
data = pd.DataFrame()

for i in range(0,6):
    gs=select_model1(x,y1,model_list[i],score=titsc)
    cv=gs.cv_results_
    a=np.array([ abs(float(cv['split0_test_score'])),abs(float(cv['split1_test_score'])),
                abs(float(cv['split2_test_score'])),abs(float(cv['split3_test_score'])),
               abs( float(cv['split4_test_score'])) ])
    n=model_list[i]
    data[n] = a 

data.to_excel('TM_comparison.xlsx', index=False)






model_list=['LR','KNN','DT','RF','SVR','MLP']
titsc='MAPE'
sc_list1=[]
#best_list1=[]
data = pd.DataFrame()

for i in range(0,6):
    gs=select_model2(x,y2,model_list[i],score=titsc)
    cv=gs.cv_results_
    a=np.array([ abs(float(cv['split0_test_score'])),abs(float(cv['split1_test_score'])),
                abs(float(cv['split2_test_score'])),abs(float(cv['split3_test_score'])),
               abs( float(cv['split4_test_score'])) ])
    n=model_list[i]
    data[n] = a 

data.to_excel('TC_comparison.xlsx', index=False)



