# Class for discretised R2C2 model

import numpy as np
from scipy.linalg import expm
from numpy.linalg import inv

import statsmodels.formula.api as smf
import statsmodels.tsa.arima_model as smarma
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

class R2C2_model():
    def __init__(self, R1, R2, C1, C2, k1, k2, delta_t):
        # Matrices of the system in continuous form
        self.Ac = np.array([[-1/(C1*R1)-1/(C1*R2), 1/(C1*R2)],
                       [1/(C2*R2), -1/(C2*R2)]])
        
        self.Bc = np.array([[1/(C1*R1), 0, k1/C1],
                       [0, 1/C2, k2/C2]])
        n = 2   # number of states
        # Matrices of the discretized state-space model
        self.F = expm(self.Ac*delta_t)
        self.G = np.dot(inv(self.Ac), np.dot(self.F-np.eye(n), self.Bc))
        self.H = np.array([[0, 1]])
    
class R2C2_onek_model():
    def __init__(self, R1, R2, C1, C2, k1, delta_t):
        # Matrices of the system in continuous form
        self.Ac = np.array([[-1/(C1*R1), 1/(C1*R1)],
                       [1/(C2*R1), -1/(C2*R1)-1/(C2*R2)]])
        
        self.Bc = np.array([[1/C1, k1, 0],
                       [0, 0, 1/(R2*C2)]])
        
        self.n_state = 2   # number of states
        
        # Matrices of the discretized state-space model
        self.F = expm(self.Ac*delta_t)
        self.G = np.dot(inv(self.Ac), np.dot(self.F-np.eye(self.n_state), self.Bc))
        self.H = np.array([[1, 0]])

class ARX_model():
    def __init__(self, features, target, train_data):
        i = 0
        for item in features:
            if i == 0:
                formula = target + ' ~ ' + features[i]
            else:
                formula = formula + ' + ' + features[i]
            i += 1
            
        print(formula)
        self.lm = smf.ols(formula=formula, data=train_data)
        
        (res_acf, res_qstat, res_pvalues) = acf(train_data[target].values, qstat=True)
        (res_pacf, res_pacf_confint) = pacf(train_data[target].values, method='ols', alpha=0.05)
        
        self.acf = res_acf
        self.acf_pvalues = res_pvalues
        self.acf_qstat = res_qstat
        self.pacf = res_pacf
        self.pacf_confint = res_pacf_confint

        
    def evaluate(self):
        res = self.lm.fit()
        self.fit_results = res
        print(self.fit_results.summary())
        
        self.coeff =  res.params[1:]
        self.intercept =  res.params[0]
        
        self.predictions = res.predict()
    
    def predict(self, test_x):
        self.predictions = np.dot(self.coeff, test_x.T) + self.intercept

class ARMA_model():
    def __init__(self, order, endog_var, exog_features, train_data):
        
        endog = train_data[endog_var]
        
        exog = train_data[exog_features]
        
        self.arma = smarma.ARMA(endog, order, exog)
        
        
    def evaluate(self):
        res = self.arma.fit()
        self.fit_results = res
        print(self.fit_results.summary())
        
        print(res.params)
        
        self.coeff =  res.params[1:]
        self.intercept =  res.params[0]
        
        self.predictions = res.predict()
    
    def predict(self, test_x):
        self.predictions = np.dot(self.coeff, test_x.T) 
        
    
        