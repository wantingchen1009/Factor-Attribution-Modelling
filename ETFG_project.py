
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import pandas_datareader as pdr
import pandas_datareader.data as web
import datetime
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold,train_test_split
from matplotlib.ticker import MaxNLocator
from sklearn import linear_model


# ## 1. Framework Setup
# ### 1.1 Data Preparation and Framework Setup

# In[4]:


# Read Fama French 3 factors daily data from website 
FF3_factors = pdr.DataReader("F-F_Research_Data_Factors_daily", "famafrench")[0]
FF3_factors.index = pd.to_datetime(FF3_factors.index, format="%Y%m%d", utc=True)

# Read Fama French 5 factors daily data from website 
FF5_factors = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_Daily", "famafrench")[0]
FF5_factors.index = pd.to_datetime(FF5_factors.index, format="%Y%m%d", utc=True)

# Set start date and end date of daily data
start = datetime.datetime(2013, 7, 31)
end = datetime.datetime(2017, 7, 31)

#Set the ETF names
ETF_names=['SPY','ACWI','EFA','EEM','IWN']

# Read 5 ETFs daily price from Yahoo 
ETF = ['SPY','ACWI','EFA','EEM','IWN']
for i in range(5):
    ETF[i] = web.DataReader(ETF[i], 'yahoo', start, end)


# In[6]:


# Set start date and end date of monthly data 
# Returns strat from 07/01/2013
start_m = datetime.datetime(2013, 6, 1)
end_m = datetime.datetime(2017, 7, 31)

# Read 5 ETFs monthly price from Yahoo 
ETF_M = ['SPY','ACWI','EFA','EEM','IWN']
for i in range(5):
    ETF_M[i] = web.get_data_yahoo(ETF_M[i],start_m,end_m,interval='m')


# In[7]:


#Set Fama French 3 factors data as the same period as ETFs
FF3_factors=FF3_factors.loc['2013-07-31':'2017-07-31']

#Set Fama French 5 factors data as the same period as ETFs
FF5_factors=FF5_factors.loc['2013-07-31':'2017-07-31']

# Calculate daily rate of return of ETFs
for i in range(5):
    Adj_Close=pd.DataFrame(ETF[i]['Adj Close'])
    Adj_Close['pct_change'] = Adj_Close.pct_change()
    ETF[i]['return']=Adj_Close['pct_change']
    
# Calculate monthly rate of return of ETFs
for i in range(5):
    Adj_Close=pd.DataFrame(ETF_M[i]['Adj Close'])
    Adj_Close['pct_change'] = Adj_Close.pct_change()
    ETF_M[i]['return']=Adj_Close['pct_change']


# In[8]:


# Detect collinearity 
def test_collinearity(df):
    cov=df.cov()
    eig_vals, eig_vecs = np.linalg.eig(cov)
    print('Eigenvectors \n%s' %eig_vecs)
    print('\nEigenvalues \n%s' %eig_vals)


# In[9]:


# Define OLS function
def OLS(Y,X):
    X = ts.add_constant(X)
    model = ts.OLS(Y,X,missing='drop')
    results = model.fit()
    print(results.summary())
    
# Define Beta function from OLS regression
def OLSBeta(Y,X):
    X = ts.add_constant(X)
    model = ts.OLS(Y,X,missing='drop')
    results = model.fit()
    return(results.params)


# In[10]:


# Define a function of ADF test 
def ADF_test(data, lag=0):
    data=data.dropna(axis=0)
    adf = ts.adfuller(data, maxlag=lag)
    print('adf: ', adf[0] )
    print('p-value: ', adf[1])
    print('critical values: ', adf[4])    
    if adf[0]> adf[4]['5%']: 
        print('Time series is nonstationary.', '\n')
        # test for 1-order difference
        print('Start testing log transformation of time series...')
        temp = np.log(data)-np.log(data.shift(1))
        temp=temp.dropna()
        adf_d = ts.adfuller(temp, maxlag=0)
        print('adf: ', adf_d[0] )
        print('p-value: ', adf_d[1])
        print('critical values: ', adf_d[4])
        if adf_d[0] > adf_d[4]['5%']: 
            print('log transformation is nonstationary.''\n')
        else:
            print('log transformation is stationary.')
    else:
        print('Time series is stationary.')


# ### 1.2 OLS Regression Using Fama-French 3 Factor Data Set

# In[11]:


# Test
print('ADF test for Fama-French 3-factors:')
print('1. ADF test of Mkt-RF:')
ADF_test(FF3_factors['Mkt-RF'])
print('\n\n2. ADF test of SMB:')
ADF_test(FF3_factors['SMB'])
print('\n\n3. ADF test of HML:')
ADF_test(FF3_factors['HML'])
print('\n\n4. ADF test of ETFs:')
for i in range(5):
    print("\n%d. ADF test of ETF: %s" %(i+1,ETF_names[i]))
    ADF_test(ETF[i]['return'])


# In[12]:


# Test collinearity
data=pd.DataFrame(FF3_factors[['Mkt-RF','SMB','HML']])
test_collinearity(data)


# In[13]:


# Make OLS regressions with Fama-French 3 factors
for i in range(5):
    print("\n%d. Regression results of ETF: %s" %(i+1,ETF_names[i]))
    Y = pd.DataFrame(ETF[i]['return'] - FF3_factors['RF'])
    X = FF3_factors[['Mkt-RF','SMB','HML']]
    OLS(Y,X)


# ## 2. Data Expansion

# ### 2.1 OLS Regression Using Fama-French 5 Factor Data Set

# In[14]:


# ADF Test
print('ADF test for Fama-French 5-factors:')
print('1. ADF test for Mkt-RF:')
ADF_test(FF5_factors['Mkt-RF'])
print('\n\n2. ADF test for SMB:')
ADF_test(FF5_factors['SMB'])
print('\n\n3. ADF test for HML:')
ADF_test(FF5_factors['HML'])
print('\n\n4. ADF test for RMW:')
ADF_test(FF5_factors['RMW'])
print('\n\n5. ADF test for CMA:')
ADF_test(FF5_factors['CMA'])


# In[15]:


# Test collinearity
data=pd.DataFrame(FF5_factors[['Mkt-RF','SMB','HML','RMW','CMA']])
test_collinearity(data)


# In[16]:


# Make OLS regressions with Fama-French 5 factors data set
for i in range(5):
    print("\n%d. Regression results of ETF: %s" %(i+1,ETF_names[i]))
    Y = pd.DataFrame(ETF[i]['return'] - FF5_factors['RF'])
    X = FF5_factors[['Mkt-RF','SMB','HML','RMW','CMA']]
    OLS(Y,X)


# ### 2.2 OLS Regression using AQR Factor Data Set

# In[17]:


# Upload AQR factor data set
AQR_BAB = pd.read_excel("Betting Against Beta Equity Factors Monthly.xlsx",sheetname='BAB Factors',skiprows=1062,skip_footer=1,names=["Date","BAB"],parse_cols="A,Z")
AQR_QMJ = pd.read_excel("Quality Minus Junk Factors Monthly.xlsx",sheetname='QMJ Factors',skiprows=1062,skip_footer=1,names=["Date","QMJ"],parse_cols="A,Z")
AQR_VME = pd.read_excel("Value and Momentum Everywhere Factors Monthly.xlsx",sheetname='VME Factors',skiprows=519,skip_footer=1,names=["Date","VALUE","MOM"],parse_cols="A,D,E")
AQR_MKT = pd.read_excel("Betting Against Beta Equity Factors Monthly.xlsx",sheetname='MKT',skiprows=1062,skip_footer=1,names=["Date","MKT"],parse_cols="A,Z")
AQR_SMB = pd.read_excel("Betting Against Beta Equity Factors Monthly.xlsx",sheetname='SMB',skiprows=1062,skip_footer=1,names=["Date","SMB"],parse_cols="A,Z")
AQR_RF = pd.read_excel("Betting Against Beta Equity Factors Monthly.xlsx",sheetname='RF',skiprows=1062,skip_footer=2,names=["Date","RF"],parse_cols="A,B")
AQR_MKT_RF=pd.DataFrame({'Date' : AQR_MKT["Date"],'MKT-RF' : AQR_MKT["MKT"]-AQR_RF["RF"]})


# In[18]:


# ADF Test
print('ADF test for AQR factor data set:')
print('1. ADF test for Mkt-RF:')
ADF_test(AQR_MKT_RF['MKT-RF'])
print('\n\n2. ADF test for SMB:')
ADF_test(AQR_SMB['SMB'])
print('\n\n3. ADF test for BAB:')
ADF_test(AQR_BAB['BAB'])
print('\n\n4. ADF test for QMJ:')
ADF_test(AQR_QMJ['QMJ'])
print('\n\n5. ADF test for VME:')
ADF_test(AQR_VME['VALUE'])
ADF_test(AQR_VME['MOM'])
# ADF Test for ETFs monthly returns
print('\n\n6. ADF test for ETFs:')
for i in range(5):
    print("\n%d. Regression results of ETF: %s" %(i+1,ETF_names[i]))
    ADF_test(ETF_M[i]['return'])


# In[19]:


# Make OLS regressions with AQR factor data set
for i in range(5):
    print("\n%d. Regression results of ETF: %s" %(i+1,ETF_names[i]))
    Y = pd.DataFrame(np.subtract(ETF_M[i]['return'][1:],AQR_RF['RF']))
    X = pd.DataFrame([AQR_MKT_RF['MKT-RF'],AQR_SMB['SMB'],AQR_BAB['BAB'],AQR_QMJ['QMJ'],AQR_VME['VALUE'],AQR_VME['MOM']]).T
    X = X.set_index(Y.index)
    OLS(Y,X)


# In[20]:


# Test collinearity
test_collinearity(X)


# Note: The condition number is 131 (higher than 30) and eigenvalues are close to zero. It is collinear which should proceed with Ridge regression to resolve the problem. The reason may be the negative correlation between value and momentum. 

# ### 2.3 OLS Regression using MSCI Factor Data Set
# Custom data by creating market neutral portfolios from a suite of MSCI long-only Indexes.  

# In[21]:


# Upload MSCI factor data set
MSCI_low_vol = pd.read_excel("MSCI.xlsx",sheetname='MSCI World Min Vol Index', skiprows=1, names=["Date","Low Vol"],parse_cols="A,B")
MSCI_world = pd.read_excel("MSCI.xlsx",sheetname='MSCI World Index', skiprows=1, names=["Date","World Index"],parse_cols="A,B")
MSCI_mom = pd.read_excel("MSCI.xlsx",sheetname='MSCI World Mom Index', skiprows=1, names=["Date","Momentum"],parse_cols="A,B")
MSCI_value = pd.read_excel("MSCI.xlsx",sheetname='MSCI World Value Weighted Index', skiprows=1, names=["Date","Value"],parse_cols="A,B")
MSCI_size = pd.read_excel("MSCI.xlsx",sheetname='MSCI World Equal Weighted Index', skiprows=1, names=["Date","Size"],parse_cols="A,B")

MSCI_index = pd.DataFrame({'Date':MSCI_low_vol['Date'], 
                           'Low Vol' : MSCI_low_vol['Low Vol'], 
                           'World Index' : MSCI_world['World Index'],
                           'Momentum': MSCI_mom['Momentum'], 
                           'Value': MSCI_value['Value'], 
                           'Size': MSCI_size['Size']})
MSCI_index = MSCI_index.set_index('Date')
MSCI_returns=MSCI_index.pct_change()
MSCI_returns['RF']=FF5_factors['RF']
MSCI_returns['Mkt-RF']=FF5_factors['Mkt-RF']
MSCI_returns=MSCI_returns.dropna()


# In[22]:


# calculate beta
Y = pd.DataFrame(MSCI_returns['Low Vol'] - MSCI_returns['RF'])
X = MSCI_returns['Mkt-RF']
Low_Vol_fit = OLSBeta(Y,X)
Beta1 = Low_Vol_fit['Mkt-RF']

Y = pd.DataFrame(MSCI_returns['World Index'] - MSCI_returns['RF'])
X = MSCI_returns['Mkt-RF']
Wor_Ind_fit = OLSBeta(Y,X)
Beta2 = Wor_Ind_fit['Mkt-RF']

w1 = Beta2/(Beta1+Beta2)
w2 = 1-w1


# In[23]:


# Define MSCI marekt netural returns 
MSCI_returns_neu = pd.DataFrame({'Date':MSCI_returns.index.values,
                                 'Adj Low Vol_neu' : MSCI_returns['Low Vol']*w1*Beta1-MSCI_returns['World Index']*w2*Beta2,
                                 'Momentum_neu': MSCI_returns['Momentum']-MSCI_returns['World Index'], 
                                 'Value_neu': MSCI_returns['Value']-MSCI_returns['World Index'], 
                                 'Size_neu': MSCI_returns['Size']-MSCI_returns['World Index'],
                                 'Mkt-RF':MSCI_returns['Mkt-RF']})
MSCI_returns_neu = MSCI_returns_neu.set_index('Date')


# In[24]:


# Test collinearity
test_collinearity(MSCI_returns_neu)

# ADF Test
print('\nADF test for AQR factor data set:')
print('1. ADF test for Mkt-RF:')
ADF_test(MSCI_returns_neu['Mkt-RF'])
print('\n\n2. ADF test for Adj Low Vol:')
ADF_test(MSCI_returns_neu['Adj Low Vol_neu'])
print('\n\n3. ADF test for Momentum:')
ADF_test(MSCI_returns_neu['Momentum_neu'])
print('\n\n4. ADF test for Value:')
ADF_test(MSCI_returns_neu['Value_neu'])
print('\n\n4. ADF test for Size:')
ADF_test(MSCI_returns_neu['Size_neu'])


# In[25]:


# Make OLS regressions with market neutral MSCI factor data set
for i in range(5):
    print("\n%d. Regression results of ETF: %s" %(i+1,ETF_names[i]))
    Y = pd.DataFrame(ETF[i]['return'][1:] - MSCI_returns['RF'])
    X = MSCI_returns_neu
    OLS(Y,X)


# Note: The results have the same problem with AQR data set which have collinearity. It needs to be resloved by Ridge Regression. 

# ## 3.Model Expansion

# ### 3.1 Polynomial
# 

# 1. Generate a new feature matrix only consisting of the squared term of each factor since the interactions have little economic implications in our model.
# 2. Use linear regression to fit to data.
# 3. Get parameters for this estimator.

# In[26]:


"""
    Define a function of polynomial regression 

    X: DataFrame of independent variables (Factors)
    Y: DataFrame of dependent variables (ETF)

    """
def poly (Y,X):
    X2 = X**2
    X2 = X2.add_suffix('^2')
    X = pd.concat([X,X2], axis=1)
    OLS(Y,X)


# #### 3.1.1  Fama French 3 Factor Data Set

# In[27]:


factor_FF3=['Mkt-RF','SMB','HML']
for i in range(5):
    print("\n%d. Polynomial regression results of ETF: %s" %(i+1,ETF_names[i]))
    Y = pd.DataFrame(ETF[i]['return'] - FF3_factors['RF'])
    X = FF3_factors[['Mkt-RF','SMB','HML']]
    poly(Y,X)


# #### 3.1.2 Fama French 5 Factor Data Set

# In[28]:


factor_FF5=['Mkt-RF','SMB','HML','RMW','CMA']
for i in range(5):
    Y = pd.DataFrame(ETF[i]['return'] - FF5_factors['RF'])
    X = FF5_factors[['Mkt-RF','SMB','HML','RMW','CMA']]
    print("\n%d. Polynomial regression results of ETF: %s" %(i+1,ETF_names[i]))
    poly(Y, X)


# #### 3.1.3 AQR Factor Data Set

# In[29]:


factor_AQR=['Mkt-RF','SMB','BAB','QMJ','VALUE','MOM']
for i in range(5):
    Y = pd.DataFrame(np.subtract(ETF_M[i]['return'][1:],AQR_RF['RF']))
    X = pd.DataFrame([AQR_MKT_RF['MKT-RF'],AQR_SMB['SMB'],AQR_BAB['BAB'],AQR_QMJ['QMJ'],AQR_VME['VALUE'],AQR_VME['MOM']]).T
    X = X.set_index(Y.index)
    print("\n%d. Polynomial regression results of ETF: %s" %(i+1,ETF_names[i]))
    poly(Y, X)


# #### 3.1.4 MSCI Factor Data Set

# In[30]:


factor_MSCI=["Adj Low Vol","Mkt-RF","Momentum","Size","Value"]
print("Note: All factors are market neutral (short MSCI World Index)")
for i in range(5):
    Y = pd.DataFrame(ETF[i]['return'][1:] - MSCI_returns['RF'])
    X = MSCI_returns_neu
    print("\n%d. Polynomial regression results of ETF: %s" %(i+1,ETF_names[i]))
    poly(Y, X)


# ### 3.2 Stepwise
# 1. Run regression with all n factors
# 2. Run n linear regressions using n-1 variables and keep the regression with highest adjusted R^2 
# 3. Run n-1 linear regressions using n-2 variables and keep the regression with highest adjusted R^2 and so forth
# 4. continue to eliminate variables and record all ajusted R^2s 
# 5. select the highest ajusted R^2 as the best regression  

# In[31]:


"""
   Define a function of stepwise regression 

   n_total: Total number of factors
   n_remains: Number of factors remained
   X: DataFrame of independent variables (Factors)
   Y: DataFrame of dependent variables (ETF)

   """
def stepwise(n_total,n_remains,X,Y):
   X_feature=X
   n_iters=len(X_feature.columns)
   Adjust_Rsquare=[]
   # Run regression of all variables
   X=ts.add_constant(X)
   model=ts.OLS(Y,X,missing='drop')
   results=model.fit()
   org_adj_R=results.rsquared_adj
   # Repeat for step of dropping 
   for j in range (0,n_total-n_remains):
       reg_score=[]
       # Drop one variable a time
       for i in range (0, n_iters):
           X1=X_feature.drop(X_feature.columns[i], axis=1) 
           X1=ts.add_constant(X1)  
           model=ts.OLS(Y,X1,missing='drop')
           results=model.fit()
           reg_score.append(results.rsquared_adj)
       # Select the variables with highest adjusted R^2
       selct=reg_score.index(max(reg_score))
       Adjust_Rsquare.append(max(reg_score))
       X_feature=X_feature.drop(X_feature.columns[selct], axis=1)  
       n_iters=n_iters-1
   Adjust_Rsquare[:0]=[org_adj_R] 
   n_index=Adjust_Rsquare.index(max(Adjust_Rsquare))
   remain=n_total-n_index
   return Adjust_Rsquare,X_feature,remain,n_index


# #### 3.2.1  Fama French 3 Factor Data Set

# In[32]:


# Implement stepwise regression
Adjust_Rsquare_FF3=[]
number_FF3=[]
for i in range(5):
    Y = pd.DataFrame(ETF[i]['return'] - FF3_factors['RF'])
    X = FF3_factors[['Mkt-RF','SMB','HML']]
    reg=stepwise(len(X.columns),1,X,Y)
    Adjust_Rsquare_FF3.append(reg[0])
    number_FF3.append(reg[2])
    print("\n%d. Regression results of ETF: %s" %(i+1,ETF_names[i]))
    print('The adjusted R^2:' )
    print(reg[0])
    print('Number of Factors remained:')
    print(reg[2])


#  The factors remained in all the 5 ETFs are market returns, size and value.

# #### 3.2.2 Fama French 5 Factor Data Set

# In[33]:


# Implement stepwise regression
Adjust_Rsquare_FF5=[]
number_FF5=[]
for i in range(5):
    Y = pd.DataFrame(ETF[i]['return'] - FF5_factors['RF'])
    X = FF5_factors[['Mkt-RF','SMB','HML','RMW','CMA']]
    reg=stepwise(len(X.columns),1,X,Y)
    Adjust_Rsquare_FF5.append(reg[0])
    number_FF5.append(reg[2])
    print("\n%d. Regression results of ETF: %s" %(i+1,ETF_names[i]))
    print('The adjusted R^2:' )
    print(reg[0])
    print('Number of Factors remained:')
    print(reg[2])


# In[34]:


# Results of regression of selected factors
for i in range(5):
    print("\n%d. Regression results of ETF: %s" %(i+1,ETF_names[i]))
    Y = pd.DataFrame(ETF[i]['return'] - FF5_factors['RF'])
    X = FF5_factors[['Mkt-RF','SMB','HML','RMW','CMA']]
    result_FF5=stepwise(len(X.columns),number_FF5[i],X,Y)
    print('The remaining factors are:')
    print(result_FF5[1].columns.tolist())
    OLS(Y,result_FF5[1])


# #### 3.2.3 AQR Factor Data Set

# In[35]:


# Implement stepwise regression
Adjust_Rsquare_AQR=[]
number_AQR=[]
for i in range(5):
    Y = pd.DataFrame(np.subtract(ETF_M[i]['return'][1:],AQR_RF['RF']))
    X = pd.DataFrame([AQR_MKT_RF['MKT-RF'],AQR_SMB['SMB'],AQR_BAB['BAB'],AQR_QMJ['QMJ'],AQR_VME['VALUE'],AQR_VME['MOM']]).T
    X = X.set_index(Y.index)
    reg=stepwise(len(X.columns),1,X,Y)
    Adjust_Rsquare_AQR.append(reg[0])
    number_AQR.append(reg[2])
    print("\n%d. Regression results of ETF: %s" %(i+1,ETF_names[i]))
    print('The adjusted R^2:' )
    print(reg[0])
    print('Number of Factors remained:')
    print(reg[2])


# In[36]:


# Results of regression of selected factors
for i in range(5):
    Y = pd.DataFrame(np.subtract(ETF_M[i]['return'][1:],AQR_RF['RF']))
    X = pd.DataFrame([AQR_MKT_RF['MKT-RF'],AQR_SMB['SMB'],AQR_BAB['BAB'],AQR_QMJ['QMJ'],AQR_VME['VALUE'],AQR_VME['MOM']]).T
    X = X.set_index(Y.index)
    result_AQR=stepwise(len(X.columns),number_AQR[i],X,Y)
    print("\n%d. Regression results of ETF: %s" %(i+1,ETF_names[i]))
    print('The remaining factors are:')
    print(result_AQR[1].columns.tolist())
    OLS(Y,result_AQR[1])


# #### 3.2.4 MSCI Factor Data Set

# In[37]:


# Implement stepwise regression
Adjust_Rsquare_MSCI=[]
number_MSCI=[]
for i in range(5):
    Y = pd.DataFrame(ETF[i]['return'][1:] - MSCI_returns['RF'])
    X = MSCI_returns_neu
    reg=stepwise(len(X.columns),1,X,Y)
    Adjust_Rsquare_MSCI.append(reg[0])
    number_MSCI.append(reg[2])
    print("\n%d. Regression results of ETF: %s" %(i+1,ETF_names[i]))
    print('The adjusted R^2:' )
    print(reg[0])
    print('Number of Factors remained:')
    print(reg[2])


# In[38]:


# Results of regression of selected factors
for i in range(5):
    Y = pd.DataFrame(ETF[i]['return'][1:] - MSCI_returns['RF'])
    X = MSCI_returns_neu
    result_MSCI=stepwise(len(X.columns),number_MSCI[i],X,Y)
    print("\n%d. Regression results of ETF: %s" %(i+1,ETF_names[i]))
    print('The remaining factors are:')
    print(result_MSCI[1].columns.tolist())
    OLS(Y,result_MSCI[1])


# ### 3.3 Lasso & Ridge

# 1. We use cross-validation technique to determine optimal alpha. 
# 2. LassoCV and RidgeCV use Least Angle Regression method.
# 3. The cross-validation techniques are included in sklearn package.

# In[39]:


"""
    Define two functions of Lasso and Ridge Regression using scikit-learn package

    X: DataFrame of independent variables (Factors)
    Y: DataFrame of dependent variables (ETF)

"""

def Lasso_L1(Y, X, factor_string):
    factor = []
    lassocv = linear_model.LassoCV(fit_intercept=True,random_state=None)
    results = lassocv.fit(X, Y.values.ravel())
    print("The optimized alpha is", results.alpha_)
    print("Coeffcients of intercept %f" %results.intercept_)
    for i in range(len(factor_string)):
        print("Coeffcients of %s : %f" %(factor_string[i],results.coef_[i])) 
        if abs(results.coef_[i] - 0) >= 0.00001:
            factor.append(factor_string[i])
    print("Score =", results.score(X,Y))
    print("Remaining factors: %s" %factor)
    return results.coef_,results.intercept_
    
def Ridge_L2(Y, X,factor_string): 
    ridgecv = linear_model.RidgeCV(fit_intercept=True)
    results = ridgecv.fit(X, Y)
    print("The optimized alpha is",ridgecv.alpha_)
    results.coef_=np.array(results.coef_[0])
    print("Coeffcients of intercept %f" %results.intercept_)
    for i in range(len(factor_string)):
        print("Coeffcients of %s : %f" %(factor_string[i],results.coef_[i]))
    print("score: ",results.score(X,Y))
    return results.coef_,results.intercept_


# #### 3.3.1 Fama French 3 Factor Data Set

# In[40]:


factor_FF3=['Mkt-RF','SMB','HML']
for i in range(5):
    Y = pd.DataFrame(ETF[i]['return'] - FF3_factors['RF'])
    X = FF3_factors[['Mkt-RF','SMB','HML']]
    Y = Y.drop(Y.index[0])
    X = X.drop(X.index[0])
    print("\n%d.1 Lasso Regression results of ETF: %s" %(i+1,ETF_names[i]))
    Lasso_L1(Y, X, factor_FF3)
    print("\n%d.2 Ridge Regression results of ETF: %s" %(i+1,ETF_names[i]))
    Ridge_L2(Y, X, factor_FF3)


# #### 3.3.2 Fama French 5 Factor Data Set

# In[41]:


factor_FF5=['Mkt-RF','SMB','HML','RMW','CMA']
for i in range(5):
    Y = pd.DataFrame(ETF[i]['return'] - FF5_factors['RF'])
    X = FF5_factors[['Mkt-RF','SMB','HML','RMW','CMA']]
    Y = Y.drop(Y.index[0])
    X = X.drop(X.index[0])
    print("\n%d.1 Lasso Regression results of ETF: %s" %(i+1,ETF_names[i]))
    Lasso_L1(Y, X, factor_FF5)
    print("\n%d.2 Ridge Regression results of ETF: %s" %(i+1,ETF_names[i]))
    Ridge_L2(Y, X, factor_FF5)


# #### 3.3.3 AQR Factor Data Set 

# In[42]:


factor_AQR=['Mkt-RF','SMB','BAB','QMJ','VALUE','MOM']
for i in range(5):
    Y = pd.DataFrame(np.subtract(ETF_M[i]['return'][1:],AQR_RF['RF']))
    X = pd.DataFrame([AQR_MKT_RF['MKT-RF'],AQR_SMB['SMB'],AQR_BAB['BAB'],AQR_QMJ['QMJ'],AQR_VME['VALUE'],AQR_VME['MOM']]).T
    Y = Y.drop(Y.index[0])
    X = X.drop(X.index[0])
    X = X.set_index(Y.index)
    print("\n%d.1 Lasso Regression results of ETF: %s" %(i+1,ETF_names[i]))
    Lasso_L1(Y, X, factor_AQR)
    print("\n%d.2 Ridge Regression results of ETF: %s" %(i+1,ETF_names[i]))
    Ridge_L2(Y, X, factor_AQR)


# #### 3.3.4 MSCI Factor Data Set

# In[51]:


factor_MSCI=["Adj Low Vol","Mkt-RF","Momentum","Size","Value"]
print("Note: All factors are market neutral (short MSCI World Index)")
for i in range(5):
    Y = pd.DataFrame(ETF[i]['return'][1:] - MSCI_returns['RF'])
    X = MSCI_returns_neu
    Y = Y.drop(Y.index[0])
    X = X.drop(X.index[0])
    print("\n%d.1 Lasso Regression results of ETF: %s" %(i+1,ETF_names[i]))
    Lasso_L1(Y, X, factor_MSCI)
    print("\n%d.2 Ridge Regression results of ETF: %s" %(i+1,ETF_names[i]))
    Ridge_L2(Y, X, factor_MSCI)


# ### 3.4 ElasticNet

# We use cross-validation technique to determine optimal alpha. 
# 1. Split the data into 0-5 folds and calculate the optimal alpha with minimun prediction error. 
# 2. Take the mean of alphas and make ElasticNet regression with whole data set.

# In[46]:


"""
   Define a function of ElasticNet Regression using scikit-learn package

   X: DataFrame of independent variables (Factors)
   Y: DataFrame of dependent variables (ETF)
   K_Fold: Number of cross-validation folds
   factor_string: Names of factors

   """
def ElasticNet_CV(X,Y,K_Fold,factor_string):
   factor = []
   Y = Y.fillna(0)
   X = X.fillna(0)
   X=np.array(X)
   Y=np.array(Y)
   Y=np.ravel(Y)
   ElasticNet_cv = ElasticNetCV(fit_intercept=True,random_state=0)
   k_fold = KFold(K_Fold)
   mean_alpha=[]
   # Print alphas and scores
   print("Alpha parameters maximising the generalization score on different subsets of the data:")
   for k, (train, test) in enumerate(k_fold.split(X, Y)):
       ElasticNet_cv.fit(X[train], Y[train])
       print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
              format(k+1, ElasticNet_cv.alpha_,ElasticNet_cv.score(X[test], Y[test])))
       mean_alpha.append(ElasticNet_cv.alpha_)
   # Averaged alpha
   mean_alpha=np.mean(mean_alpha)
   regr = ElasticNet(alpha=mean_alpha,fit_intercept=True, random_state=0)
   result = regr.fit(X, Y)
   print("Mean alpha: %f" %mean_alpha)
   print("\nIntercept: %f" %result.intercept_)
   for i in range(len(factor_string)):
       print("Coeffcients of %s : %f" %(factor_string[i],result.coef_[i]))
       if abs(result.coef_[i]-0) >= 0.00001:
           factor.append(factor_string[i])
   N = X.shape[0]
   k = X.shape[1]
   score = result.score(X,Y)
   print("Score: ",score)
   print("Remaining factors: %s" %factor)
   return result.intercept_,result.coef_,mean_alpha


# #### 3.4.1 Fama French 3 Factor Data Set

# In[47]:


factor_FF3=['Mkt-RF','SMB','HML']
res_FF3=[]
for i in range(5):
    Y = pd.DataFrame(ETF[i]['return'] - FF3_factors['RF'])
    X = FF3_factors[['Mkt-RF','SMB','HML']]
    print("\n%d. ElasticNet regression results of ETF: %s" %(i+1,ETF_names[i]))
    res_FF3.append(ElasticNet_CV(X,Y,5,factor_FF3))


# #### 3.4.2 Fama French 5 Factor Data Set

# In[48]:


factor_FF5=['Mkt-RF','SMB','HML','RMW','CMA']
res_FF5=[]
for i in range(5):
    Y = pd.DataFrame(ETF[i]['return'] - FF5_factors['RF'])
    X = FF5_factors[['Mkt-RF','SMB','HML','RMW','CMA']]
    print("\n%d. ElasticNet regression results of ETF: %s" %(i+1,ETF_names[i]))
    res_FF5.append(ElasticNet_CV(X,Y,5,factor_FF5))


# #### 3.4.3 AQR Factor Data Set 

# In[49]:


factor_AQR=['Mkt-RF','SMB','BAB','QMJ','VALUE','MOM']
res_AQR=[]
for i in range(5):
    Y = pd.DataFrame(np.subtract(ETF_M[i]['return'][1:],AQR_RF['RF']))
    X = pd.DataFrame([AQR_MKT_RF['MKT-RF'],AQR_SMB['SMB'],AQR_BAB['BAB'],AQR_QMJ['QMJ'],AQR_VME['VALUE'],AQR_VME['MOM']]).T
    X = X.set_index(Y.index)
    print("\n%d. ElasticNet regression results of ETF: %s" %(i+1,ETF_names[i]))
    res_AQR.append(ElasticNet_CV(X,Y,5,factor_AQR))


# #### 3.4.4 MSCI Factor Data Set

# In[50]:


factor_MSCI=["Adj Low Vol","Mkt-RF","Momentum","Size","Value"]
res_MSCI=[]
print("Note: All factors are market neutral (short MSCI World Index)")
for i in range(5):
    Y = pd.DataFrame(ETF[i]['return'][1:] - MSCI_returns['RF'])
    X = MSCI_returns_neu
    print("\n%d. ElasticNet regression results of ETF: %s" %(i+1,ETF_names[i]))
    res_MSCI.append(ElasticNet_CV(X,Y,5,factor_MSCI))

