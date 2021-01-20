
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model 
from sklearn import preprocessing

import pickle 

df=pd.read_csv('/home/pc/machinelearningdatasets/indianationalincome1970.csv')
ndf=pd.read_csv('/home/pc/machinelearningdatasets/predictyear.csv')
print(df)

poly=preprocessing.PolynomialFeatures(3)
Year_t=poly.fit_transform(df[['Year']])
Yearp_t=poly.fit_transform(ndf[['PredictYear']])

reg=linear_model.LinearRegression()


reg.fit(Year_t,df.Income)



plt.xlabel("Financial - Year")
plt.ylabel("PerCapitaIncome( US $ )")
plt.title(" India PerCapitaIncome( US $ ) ")
plt.scatter(df.Year,df.Income,color='red')
plt.plot(df.Year,reg.predict(Year_t),color='blue')
plt.show()
with open('model_pickle','wb') as f:
	pickle.dump(reg,f) 
with open('model_pickle','rb') as f:
	np=pickle.load(f)

inptyr=int(input('Enter year to see expected PerCapitaIncome(US $) '))

print("Expected PerCapitaIncome( US $ ) in year", inptyr,' as per our model',np.predict(poly.fit_transform([[inptyr]]) ) )
