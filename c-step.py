import numpy as np
from scipy import stats
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def cstep(data,old_set,old_model,h):
    '''data: 2-dimenional n*(p+1) array;
       old_set: the index of previous subset;
       old_cpef: previous regression model coefficients
    '''
    #first compute residual from previous model
    n=data.shape[0]
    covariates=data[:,:-1]
    obs=data[:,-1]
    prediction=old_model.predict(covariates).reshape(n)
    residual=abs(obs-prediction)

    #find the h data points with least residual
    new_set=np.argpartition(residual,h)[:h]

    #fit the new model to the new subset
    new_covariates=data[new_set,:-1]
    new_obs=data[new_set,-1]
    regr = linear_model.LinearRegression()
    regr.fit(new_covariates, new_obs)

    return((new_set,regr))


###demo
#--------------------------------------------------------------------------------
# X=100*np.random.rand(100,1)
# Y=X+np.random.normal(0,10,size=(100,1))
#
# #add outlier
# out_num=30
# out_X=90+10*np.random.rand(out_num,1)
# out_Y=out_X*10+0.5+np.random.normal(0,10,size=(out_num,1))
#
# X=np.concatenate((X,out_X),axis=0)
# Y=np.concatenate((Y,out_Y),axis=0)
#
# data=np.concatenate((X,Y),axis=1)
#
# #plot
# plt.plot(X,Y,'ro')
# plt.grid(True)
#
# #first time regression
# n,p= X.shape
# p+=1
# h=(n+p+1)/2
# model=linear_model.LinearRegression()
#
# #initial subset selection
# subset=np.random.choice(range(n),h,replace=False)
# model.fit(X[subset,:],Y[subset,:])
#
# #plot initial fitted model
# plt.plot(X,model.predict(X),ls='-',color='blue')
#
# #iteration of c-step
# newset, new_model=cstep(data,subset,model,h)
# plt.plot(X,new_model.predict(X),ls='-',color='green')
# print newset
#
# plt.show()
#
#--------------------------------------------------------------------------------

def basicLTS(data):
    '''
        computing least trimmed square estimator
        data: n*p dimension array
    '''
    n,p=data.shape
    h=(n+p+1)/2
    print h
    X=data[:,:-1]
    Y=data[:,-1]
    Y=Y[:,np.newaxis]
    model=linear_model.LinearRegression()

    #initial subset selection
    subset=np.random.choice(range(n),h,replace=False)
    model.fit(X[subset,:],Y[subset,:])

    #begin iteration of c-step, until process converge
    iter_times=1
    while True:
        newset, new_model=cstep(data,subset,model,h)
        iter_times+=1
        if all(np.sort(newset)==np.sort(subset)):
            break
        subset=newset
        model=new_model

    #plot data points and final result
    plt.plot(X,Y,'ro')
    plt.plot(X,model.predict(X),ls='-',color='blue')

    print 'Number of iteration: %d'%iter_times
    plt.show()

in_num=800
X=np.random.normal(0,100,size=(in_num,1))
Y=X+1+np.random.normal(0,1,size=(in_num,1))

#add outlier
out_num=200
out_X=np.random.normal(200,25,size=(out_num,1))
out_Y=np.random.normal(0,25,size=(out_num,1))

X=np.concatenate((X,out_X),axis=0)
Y=np.concatenate((Y,out_Y),axis=0)

data=np.concatenate((X,Y),axis=1)

basicLTS(data)
