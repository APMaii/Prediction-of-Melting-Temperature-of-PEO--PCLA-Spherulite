# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 11:21:45 2023

@author: sunhouse

Check code for possibility of modeling or not
criteria: some  criteria percentage for metrics of ML models


this is the first, after that we could do that in the goodway

"""



import numpy as np

x=np.array(([100,0],[80,20],[50,50],[20,80],[0,100]))

#y1=np.array(([66,52],[65,54],[64,55],[61,51],[61,40]))
y1=np.array([66,65,64,61,61])
y2=np.array((52,54,55,51,40))
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

fold=KFold(n_splits=5,shuffle=True,random_state=0)

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
scale= MinMaxScaler()
scale.fit(x)
x_scaled=scale.transform(x)
model11=SVR(kernel='linear')
cv_score=cross_val_score(model11,x_scaled,y1,cv=fold,scoring='neg_mean_absolute_percentage_error')

print('our score for Tm is:',cv_score.mean()) #our score for Tm is: -0.02544959772765446
#====================

model12=SVR(kernel='poly',degree=2)

cv_score=cross_val_score(model12,x_scaled,y2,cv=fold,scoring='neg_mean_absolute_percentage_error')



print('our score in TC is',cv_score.mean()) #-0.053238470530894796



import matplotlib.pyplot as plt
n2=np.arange(0,101).reshape(-1,1)
n1=np.flip(n2).reshape(-1,1)

new=np.concatenate([n1,n2],axis=1)
scaled_new=scale.transform(new)

model11.fit(x_scaled,y1)
model11_pred=model11.predict(scaled_new)
#model1_pred=model1.predict(new)


model12.fit(x_scaled,y2)
model12_pred=model12.predict(scaled_new)
#model2_pred=model2.predict(new)




from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor



model21=MLPRegressor(hidden_layer_sizes=(100,100),solver='adam',random_state=0)
model31=RandomForestRegressor(random_state=0,n_estimators=20,max_depth=4)


model21.fit(x_scaled,y1)
model21_pred=model21.predict(scaled_new)



model31.fit(x_scaled,y1)
model31_pred=model31.predict(scaled_new)




plt.scatter(x[:,1],y1,marker='*',s=100,c='r',label='experimental')
plt.plot(n2,model11_pred,c='b',label='SVR Predicted')
plt.plot(n2,model21_pred,c='g',label='MLP Predicted')
plt.plot(n2,model31_pred,c='k',label='RF Predicted')


plt.title('First Model for Tm')
plt.xlabel('PEO Concentration (%)')
plt.ylabel('Tm')
plt.legend()
plt.xlim(0,120)
#plt.ylim(30,80)
plt.ylim(50,70)

plt.show()




model21.fit(x_scaled,y2)
model22_pred=model21.predict(scaled_new)


model31.fit(x_scaled,y2)
model32_pred=model31.predict(scaled_new)


plt.scatter(x[:,1],y2,marker='*',s=100,c='r',label='experimental')
plt.plot(n2,model12_pred,c='b',label='ML Predicted')
plt.plot(n2,model22_pred,c='g',label='MLP Predicted')
plt.plot(n2,model32_pred,c='k',label='RF Predicted')

plt.title('Second Model for Tc')
plt.xlabel('PEO Concentration (%)')
plt.ylabel('Tc')
plt.legend()
plt.xlim(0,120)
#plt.ylim(30,80)
plt.ylim(40,70)
plt.show()



cv_score2=cross_val_score(model21,x_scaled,y1,cv=fold,scoring='neg_mean_absolute_percentage_error')

cv_score3=cross_val_score(model31,x_scaled,y1,cv=fold,scoring='neg_mean_absolute_percentage_error')


print('MLP our score in TM is',cv_score2.mean()) #-0.9204338307033311
print('RF our score in TM is',cv_score3.mean()) #-0.01116803278688525


cv_score2=cross_val_score(model21,x_scaled,y1,cv=fold,scoring='neg_mean_absolute_percentage_error')

cv_score3=cross_val_score(model31,x_scaled,y2,cv=fold,scoring='neg_mean_absolute_percentage_error')



print('MLP our score in Tc is',cv_score2.mean()) #-0.9204338307033311
print('RF our score in Tc is',cv_score3.mean()) #-0.011168032786885258

