#!/usr/bin/env python
"""
Analysis and Imputation of Missing Values

Copyright (c) 2022, A.A. Bessonov (bestallv@mail.ru)

Routines in this module:

aggr(data,figsize=(20, 7),title=None, barV=True,resMV=True)
bar_miss(data=None,columns=None,typeObs=None,figsize=(15,8),title=None,xlabel=None,ylabel=None)
chols(d,theta,p,psi,mc,nmc)
em_norm(s,start=None,showits=True,maxits=10000,criterion=.0001)
gauss(it)
getparam_norm(s,theta,corr=None)
gtmc(p,npatt,r,patt)
gtoc(p,npatt,r,patt)
hist_miss(df,figsize=(15,8),title=None,xlabel=None,ylabel=None)
imp_norm(s,theta,x)
initn(d)
is1n(d,theta,t,tobs,p,psi,n,x,npatt,r,mdpst,nmdp,c)
mcar_test(data,maxits=10000,criterion=.0001,alpha=.05)
object_miss(df,nORp=0.2)
prelim_norm(x)
rangen(init)
sigex(d,theta,extr,p,psi,mc,nmc)
swp(d,theta_,pivot,p,psi,submat,diR)
swpobs(d,theta_1,p,psi,npatt,r,patt)
tobsn(d,p,psi,n,x,npatt,r,mdpst,nmdp)

"""
from __future__ import division, absolute_import, print_function

__all__ = ['aggr', 'bar_miss', 'chols', 'em_norm', 'gauss', 'getparam_norm', 'gtmc',
	   'gtoc', 'hist_miss', 'imp_norm', 'initn', 'is1n', 'mcar_test', 'object_miss', 
	   'prelim_norm', 'rangen', 'sigex', 'swp', 'swpobs', 'tobsn']

import numpy as np 
import pandas as pd
from scipy import stats
import math
import sys
import time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def aggr(data,figsize=(20, 7),title=None, barV=True,resMV=True):
    """
    Calculate or plot the amount of missing values in each variable and the amount of missing values in certain combinations of variables.

    Parameters
    ----------
    data : [DataFrame] dataframe to be verified for missing values.
    figsize : [int,int] a method used to change the dimension of plot window, width, height in inches (default: figsize=(20,7)).
    title : [str] set a title for the heatmap (default: title=None).
    barV : [bool] if True (default) assign the bar values to each bar.
    resMV : [bool] if True (default) a table of variables with missed values and their count is returned.

    Returns
    ----------
    Plot the amount of missing values in each variable and plot the amount of missing values in certain combinations of variables. Table of variables with missed values and their count
    """
    nNA=data.isnull().sum()
    res=nNA[nNA!=0]
    cn=data.columns
    tmp=np.where(data.isna(),1,0)
    tmp = pd.DataFrame(tmp,columns=cn)
    tab = tmp.groupby(tmp.columns.tolist()).size().reset_index().rename(columns={0:'count'})
    tab=tab.sort_values(by=['count'])
    tab = tab.set_index('count')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    sns.barplot(y=nNA.values, x=nNA.index, color='red',ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(),rotation = 90)
    ax1.set_ylabel('Proportion of missings')
    ax1.set_xlabel('Feature')
    if barV==True:
        ax1.bar_label(ax1.containers[0])
    sns.heatmap(tab, cmap='coolwarm', linewidths=0.5, cbar=False)
    ax2.set(xlabel='Feature',ylabel='Count of objects')
    if title==None:
        plt.title("Missing values in objects and feature", size=14)
    else:
        plt.title(title, size=14)
    plt.show()
    if resMV==True:
        return res


def bar_miss(data=None,columns=None,typeObs=None,figsize=(15,8),title=None,xlabel=None,ylabel=None):
    """
    Barplot with highlighting of missing values in other variable for the studied variable.

    Parameters
    ----------
    data : [DataFrame] dataframe to be verified for missing values.
    columns : [list, str] the names of columns (features) in the data to study the ratio of observed and missed values.
    typeObs : [str] a type of values on x-axis: float (typeObs=None, default) or int (typeObs='int'). 
    figsize : [int,int] a method used to change the dimension of plot window, width, height in inches (default: figsize=(15,8)).
    title : [str] set a title for the plot (default: title=None).
    xlabel : [str] set a title for the x-axis (default: xlabel=None).
    ylabel : [str] set a title for the y-axis (default: ylabel=None).

    Returns
    ----------

    Barplot with highlighting of missing values in other variable, which is interconnected with the studied variable, by splitting each bar into two parts.
    """
    column=columns 
    nCount = data[column[0]].value_counts().rename_axis(column[0]).reset_index(name='Count')
    nCount=nCount.sort_values(by=column[0]).reset_index(drop=True)
    listUnique=list(set(data[column[0]].unique()))
    listUnique=[x for x in listUnique if x == x]
    resf=[]
    for num in range(0,len(listUnique)):
        res=len(data[(data[column[1]].isnull()) &(data[column[0]]==(listUnique[num]))])
        resf.append(res)
    dfmc=pd.DataFrame({f'{column[0]}':listUnique,f'{nCount.columns[1]}':nCount[nCount.columns[1]],f'{column[1]}':resf})
    if typeObs=='int':
        dfmc[column] = dfmc[column].applymap(np.int64)
    columnMc=dfmc.columns
    plt.figure(figsize = figsize)
    ax = sns.barplot(y=dfmc[columnMc[1]], x=dfmc[columnMc[0]], color='blue', label='existing values')
    ax = sns.barplot(y=dfmc[columnMc[2]], x=dfmc[columnMc[0]], color='red', label='missed values')
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    if title==None:
        plt.title("Missing values in observations", size=14)
    else:
        plt.title(title, size=14)
    if xlabel==None:
        plt.xlabel(columnMc[0],size=13)
    else:
        plt.xlabel(xlabel,size=13)
    if ylabel==None:
        plt.ylabel(f'missing/observed in {columnMc[2]}',size=13)
    else:
        plt.ylabel(ylabel,size=13)
    plt.legend()
    plt.show()


def chols(d,theta,p,psi,mc,nmc):
    """
    Return submatrix of theta corresponding to the columns of mc.
    """
    theta=theta.copy()
    for i in range(1,nmc):
        tmp=float(0)
        for k in range(1,i-1):
            tmp=tmp+theta[(psi[mc[k],mc[i]])-1]**2
        theta[(psi[mc[i],mc[i]])-1]=np.sqrt((theta[(psi[mc[i],mc[i]])-1])-tmp)
        theta=theta.copy()
        for j in range(i+1,nmc):
            tmp=float(0)
            for k in range(1,i-1):
                tmp=tmp+theta[(psi[mc[k],mc[i]])-1]*theta[(psi[mc[k],mc[j]])-1]
                tmp=tmp.copy()
            theta[(psi[mc[i],mc[j]])-1]=((theta[(psi[mc[i],mc[j]])-1])-tmp)/theta[(psi[mc[i],mc[i]])-1]
    return theta


def em_norm(s,start=None,maxits=10000,criterion=.0001,showits=None):
    """
    Performs maximum-likelihood estimation on the matrix of incomplete data using the EM algorithm.

    Parameters
    ----------
    s : [list, array] summary list of an incomplete normal data matrix produced by the function prelim_norm.
    start : [list] optional starting value of the parameter. If no starting value is supplied, 
                   em_norm chooses its own starting value (start=None).
    maxits : [int] maximum number of iterations performed. The algorithm will stop if the parameter still
                   has not converged after this many iterations (default maxits=10000).
    criterion : [float] convergence criterion. The algorithm stops when the maximum relative difference 
                    in all of the estimated means, variances, or covariances from one iteration to the 
                    next is less than or equal to this value (default criterion=.0001).
    showits : [bool] if True, reports the iterations of EM so the user can monitor the progress of 
                     the algorithm (default showits=None).

    Returns
    ----------
    a vector representing the maximum-likelihood estimates of the normal parameters. This vector 
    contains means, variances, and covariances on the transformed scale in packed storage.
    The parameter can be transformed back to the original scale and put into a more 
    understandable format by the function getparam_norm.
    """
    def emn(d,theta,t,tobs,p,psi,n,x,npatt,r,mdpst,nmdp):
        """
        Performs one step of EM. Theta must be in sweep(0) condition. After execution, 
        the new parameter value is contained in t, and theta is left swept on columns 
        corresponding to observed variables in the first missingness pattern.
        """
        theta=theta.copy()
        t=t.copy()
        tobs=tobs.copy()
        c=[0]*p
        for i in range(1,d):
            t[i]=tobs[i]
        for patt in range(npatt):
            theta=swpobs(d,theta,p,psi,npatt,r,patt)
            theta=theta.copy()
            mc,nmc=gtmc(p,npatt,r,patt)
            oc,noc=gtoc(p,npatt,r,patt)
            for i in range(mdpst[patt],mdpst[patt]+nmdp[patt]):
                for j in range(1,nmc):
                    c[mc[j]]=theta[(psi[0,mc[j]])-1]
                    c=c.copy()
                    for k in range(1,noc):
                        c[mc[j]]=c[mc[j]]+theta[(psi[oc[k],mc[j]])-1]*x.iloc[i,oc[k]-1]
                        c=c.copy()
                for j in range(1,nmc):
                    t[(psi[0,mc[j]])-1]=t[(psi[0,mc[j]])-1]+c[mc[j]]
                    t=t.copy()
                    for k in range(1,noc):
                        t[(psi[oc[k],mc[j]])-1]=t[(psi[oc[k],mc[j]])-1]+x.iloc[i,oc[k]-1]*c[mc[j]]
                        t=t.copy()
                    for k in range(j,nmc):
                        t[(psi[mc[k],mc[j]])-1]=t[(psi[mc[k],mc[j]])-1]+c[mc[k]]*c[mc[j]]+theta[(psi[mc[k],mc[j]])-1]
                        t=t.copy()
        for i in (range(1,d)):               
            t[i]=t[i]/n
        res_=swp(d,t,0,p,psi,p,1)
        res={'d':d, 'theta':theta, 't':t, 'tobs':tobs, 'p':p, 'psi':psi, 'n':n,
            'x':x, 'npatt':npatt, 'r':r, 'mdpst':mdpst, 'nmdp':nmdp, 'oc':oc, 'mc':mc, 'res_':res_}
        return res
    def stvaln(d,p,psi):
        """
        Gets starting value of theta: mu=0 and sigma=1.
        """
        theta=[0]*d
        for i in range(2,d):
            theta[i]=0
        theta[0]=-1
        for j in range(1,p+1):
            theta[psi[j,j]-1]=1
        return theta
    def progress(it,total,buffer=30):
        """
        A progress bar is used to display the progress of a long running Iterations of EM,
        providing a visual cue that processing is underway.
        """
        percent = 100.0*it/(total+1)
        sys.stdout.write('\r')
        sys.stdout.write("Iterations of EM: [\033[34m{:{}}] {:>3}% ".format('█'*int(percent/(100.0/buffer)),buffer, int(percent)))
        sys.stdout.flush()
        time.sleep(0.001)
    # replacing missed values by 999
    sx=s['x'].fillna(999)
    # optional starting value of the parameter
    if start==None:
        start=stvaln(s['d'],s['p'],s['psi'])
    # generation the sscp matrix for all missingness patterns
    tobs=tobsn(s['d'],s['p'],s['psi'],s['n'],sx,s['npatt'],s['r'],s['mdpst'],s['nmdp'])
    # iterate to mle
    it=0
    converged=False
    while not(converged) and it<maxits:
        if isinstance(start, list):
            old=start.copy()
        if isinstance(start, dict):
            old = [*start['t']]
            start = [*old]
        start=emn(s['d'],old,start,tobs,s['p'],s['psi'],s['n'],sx,s['npatt'],s['r'],s['mdpst'],s['nmdp'])
        it=it+1
        if showits==True:
            progress(it+1, maxits)
        converged=np.max([abs(x - y) for x, y in zip(list(start['t']),list(old))])<=criterion
    return start['t']


def gauss(it):
    """
    Generates random numbers from a Gaussian distribution.
    """
    pi=3.141593
    if it%2==0:
        u1=rangen(it+1)
        u2=rangen(it+1)
        gauss=math.sqrt(-2*math.log(u1))*math.cos(2*pi*u2)
    else:
        u1=rangen(it+1)
        u2=rangen(it+1)
        gauss=math.sqrt(-2*math.log(u1))*math.sin(2*pi*u2)
    return gauss


def getparam_norm(s,theta,corr=None):
    """
    Takes a parameter vector, such as one produced by em_norm and returns 
    a list of parameters on the original scale.

    Parameters
    ----------
    s : [list, array] summary list of an incomplete normal data matrix produced by the function prelim_norm.
    theta : [list] vector of normal parameters expressed on transformed scale in packed storage, 
                    such as one produced by the function em_norm.
    corr : [bool] if True, computes means, standard deviations, and a correlation matrix. If None (default),
                    computes means and a covariance matrix.

    Returns
    ----------
    if corr=None, a list containing the components mu and sigma;
    if corr=True, a list containing the components mu, sdv, and r.
    The components are:
    mu :    vector of means values. Elements are in the same order and on the same scale 
            as the columns of the original data matrix, and with names corresponding 
            to the column names of the original data matrix.
    sigma : matrix of variances and covariances.
    sdv :   vector of standard deviations.
    r :     matrix of correlations.
    """
    mu=np.add(np.multiply(theta[s['psi'][0][0]:s['psi'][0][s['p']]],s['sdv']),s['xbar'])
    sigma=[theta[x] for x in list(map(lambda x: x - 1, s['psi'][1:s['p']+1,1:s['p']+1].ravel()))]
    sigma=np.array(sigma).reshape(s['p'],s['p'])
    tmp=np.repeat(s['sdv'],s['p']).reshape(s['p'],s['p'])
    sigma=sigma*tmp*tmp.T
    if corr==True:
        sdv=np.sqrt(np.diagonal(sigma))
        tmp=np.repeat(sdv,s['p']).reshape(s['p'],s['p'])
        r=sigma/(tmp*tmp.T)
        result={'mu':mu,'sdv':sdv,'r':r}
    else:
        result={'mu':mu,'sigma':sigma}
    return result


def gtmc(p,npatt,r,patt):
    """
    Finds the column numbers of the missing variables, and stores them 
    in the first nmc elements of mc. Does not go beyond last column.
    """
    last=p
    nmc=0
    mc=[0]*(p+1)
    for j in range(last):
        if r.iloc[patt,j]==0:
            nmc=nmc+1
            mc[nmc]=j
    mc.pop(0)
    mc=[1 if x==0 else x for x in mc]
    return mc, nmc

def gtoc(p,npatt,r,patt):
    """
    Finds the column numbers of the observed variables, and stores them 
    in the first noc elements of oc. Does not go beyond last column.
    """
    last=p
    noc=0
    oc=[0]*(p+1)
    for i in range(last):
        if r.iloc[patt,i]==1:
            noc=noc+1
            oc[noc]=i
    oc.pop(0)
    oc = [x+1 for x in oc]
    return oc, noc


def hist_miss(df,figsize=(15,8),title=None,xlabel=None,ylabel=None):
    """
    Histogram with highlighting of missing values in variables by splitting each bin into two parts.

    Parameters
    ----------
    df : [DataFrame] dataframe to be verified for missing values.
    figsize : [int,int] a method used to change the dimension of plot window, width, height in inches (default: figsize=(15,8)).
    title : [str] set a title for the plot (default: title=None).
    xlabel : [str] set a title for the x-axis (default: xlabel=None).
    ylabel : [str] set a title for the y-axis (default: ylabel=None).

    Returns
    ----------
    Histogram with highlighting of missing values in variables: the observed values blue and missing values red
    """
    notnull=df.notnull().sum()
    null=df.isnull().sum()
    plt.figure(figsize = figsize)
    ax = sns.barplot(y=notnull.values, x=notnull.index, color='blue', label='existing values')
    ax = sns.barplot(y=null.values, x=null.index, color='red', label='missing values')
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    if title==None:
        plt.title("Missing values in signs", size=14)
    else:
        plt.title(title, size=14)
    if xlabel==None:
        plt.xlabel('Variables',size=13)
    else:
        plt.xlabel(xlabel,size=13)
    if ylabel==None:
        plt.ylabel('The number of observations',size=13)
    else:
        plt.ylabel(ylabel,size=13)   
    plt.legend()
    plt.show()


def imp_norm(s,theta,x):
    """
    Draws missing elements of a data matrix under the multivariate normal 
    model and a user-supplied parameter.

    Parameters
    ----------
    s : [list, array] summary list of an incomplete normal data matrix produced by the function prelim_norm.
    theta : [list] vector of normal parameters expressed on transformed scale in packed storage, 
                   such as one produced by the function em_norm.
    x : [DataFrame] data frame containing missing values. The rows of x correspond to observational
                    units, and the columns to variables. Missing values are denoted by NA.

    Returns
    ----------
    a matrix of the same form as x, but with all missing values filled in with simulated values 
    drawn from their predictive distribution given the observed data and the specified parameter.
    """
    x=x.copy()
    theta=theta.copy()
    sx=s['x'].fillna(999)
    tmpi=[0]*s['p']
    tmpr=[0]*s['p']
    tobs=tobsn(s['d'],s['p'],s['psi'],s['n'],sx,s['npatt'],s['r'],s['mdpst'],s['nmdp'])
    xX=is1n(s['d'],theta,theta,tobs,s['p'],s['psi'],s['n'],sx,s['npatt'],s['r'],s['mdpst'],s['nmdp'],theta)
    dfxx=xX['x'].to_numpy()*np.full((s['n'],s['p']),s['sdv'])+np.full((s['n'],s['p']),s['xbar'])
    df=pd.DataFrame(dfxx,columns=x.columns).round(1)
    df=df.loc[df.index[s['ro']]]
    df=df.reset_index(drop=True,inplace=False)
    x[x.isnull()]=df
    return x


def initn(d):
    """
    Initializes theta.
    """
    tobs=[0]*d
    tobs[0]=1
    for i in range(2,d):
        tobs[i]=0
    return tobs


def is1n(d,theta,t,tobs,p,psi,n,x,npatt,r,mdpst,nmdp,c):
    """
    Performs I-step of data augmentation. Randomly draws missing data Xmis 
    given theta, and stores the sufficient statistics in t. Theta must be 
    in sweep(0) condition. Answer is returned in unswept condition.
    """
    theta=theta.copy()
    t=t.copy()
    tobs=tobs.copy()
    z=[0]*p
    for i in range(d):
        t[i]=tobs[i]
        for patt in range(npatt-1,-1,-1):
           theta=swpobs(d,theta,p,psi,npatt,r,patt)
           theta=theta.copy()
           mc,nmc=gtmc(p,npatt,r,patt)
           oc,noc=gtoc(p,npatt,r,patt)
           theta=sigex(d,theta,c,p,psi,mc,nmc)
           theta=theta.copy()
           theta=chols(d,c,p,psi,mc,nmc)
           theta=theta.copy()
           for i in range(mdpst[patt],mdpst[patt]+nmdp[patt]):
               for j in range(nmc):
                   x.iloc[i,mc[j]]=theta[(psi[0,mc[j]])-1]
                   x=x.copy()
                   for k in range(noc):
                       x.iloc[i,mc[j]]=x.iloc[i,mc[j]]+theta[(psi[oc[k],mc[j]])-1]*x.iloc[i,oc[k]-1]
                   z[mc[j]]=gauss(j)
                   for k in range(1,j):
                       x.iloc[i,mc[j]]=x.iloc[i,mc[j]]+z[mc[k]]*c[(psi[mc[j],mc[k]])-1]
                   t[(psi[0,mc[j]])-1]=t[(psi[0,mc[j]])-1]+x.iloc[i,mc[j]-1]
                   for k in range(1,noc):
                       t[(psi[mc[j],oc[k]])-1]=t[(psi[mc[j],oc[k]])-1]+x.iloc[i,mc[j]-1]*x.iloc[i,oc[k]-1]
                       t=t.copy()
                   for k in range(1,j):
                       t[(psi[mc[j],mc[k]])-1]=t[(psi[mc[j],mc[k]])-1]+x.iloc[i,mc[j]-1]*x.iloc[i,mc[k]-1]
                       t=t.copy()
    for i in range(1,d):
        t[i]=t[i]/n
    res={'x':x,'d':d,'t':t,'theta':theta,'n':n,'p':p,'r':r,'mdpst':mdpst,
        'nmdp':nmdp, 'npatt':npatt, 'psi':psi, 
        'mc':mc, 'nmc':nmc, 'oc':oc, 'noc':noc}            
    return res


def mcar_test(data,maxits=10000,criterion=.0001,alpha=.05,showits=None):
    """
    Use Little’s (1988) test statistic to assess if data is missing completely at random (MCAR).

    Parameters
    ----------
    data : [DataFrame, Series] dataframe or vector to be tested of missing values completely at random.
    maxits : [int] maximum number of iterations performed. The algorithm will stop if the parameter still
                   has not converged after this many iterations. (default maxits=10000).
    criterion : [float] convergence criterion. The algorithm stops when the maximum relative difference 
                   in all of the estimated means, variances, or covariances from one iteration to the 
                   next is less than or equal to this value. (default criterion=.0001).
    alpha : [float] the confidence level required (default alpha=.05).
    showits : [bool] if True, reports the iterations of EM so the user can monitor the progress of 
                     the algorithm (default showits=None).

    Returns
    ----------
    A table with one row and five columns:
    statistic : Chi-squared statistic for Little’s test
    df : Degrees of freedom used for chi-squared statistic
    p.value : P-value for the chi-squared statistic 
    missing.patterns : Number of missing data patterns in the data
    missing.data : Interpretation of the test results
    """
    data=data.copy()
    # define variables
    n_var=data.shape[1]
    n=data.shape[0]
    var_names=list(data.columns)
    r=pd.DataFrame(data=np.where(data.isna(),1,0),columns=var_names)
    #missing data patterns
    mdp=np.matmul(r,[2 ** num for num in list(range(0,n_var))])+1
    x_miss_pattern = pd.concat([data, mdp], axis=1)
    x_miss_pattern.rename(columns={0:'miss_pattern'}, inplace=True)
    n_miss_pattern=len(x_miss_pattern.miss_pattern.unique())
    x_miss_pattern['miss_pattern']=x_miss_pattern['miss_pattern'].rank(method='dense').astype('int')
    s=prelim_norm(data)
    ll=em_norm(s,maxits=maxits,criterion=criterion,showits=showits)
    fit=getparam_norm(s,ll)
    grand_mean=fit['mu']
    grand_cov=fit['sigma']
    grand_cov=pd.DataFrame(data=grand_cov,columns=var_names)
    list_ind=grand_cov.index
    list_key=grand_cov.columns.to_list()
    dict_ind = {list_ind[i]: list_key[i] for i in range(len(list_ind))}
    grand_cov=grand_cov.rename(dict_ind,axis=0)
    # data
    dict_list=list(range(1,max(x_miss_pattern['miss_pattern']+1)))
    data_dict = {}
    for i in dict_list:
        data_dict[i]=x_miss_pattern[x_miss_pattern['miss_pattern']==i]
    # kj
    kj_dict = {}
    for i in dict_list:
        kj_dict[i]=max(data_dict[i].notna().sum(axis=1)-1)
    # mu
    x_miss=x_miss_pattern.groupby(by=["miss_pattern"]).mean()-grand_mean
    x_miss = x_miss.rename_axis(index=None)
    mu_dict = {}
    for i in dict_list:
        mu_dict[i]=x_miss.loc[i].dropna()
    # sigma
    sigma_dict = {}
    for i in dict_list:
        sigma_dict[i]=grand_cov.loc[mu_dict[i].index.to_list()][mu_dict[i].index.to_list()]
    # d2
    cnrow=[]
    for i in dict_list:
        if kj_dict[i]==0:
            nrow=kj_dict[i]
        else:
            nrow=data_dict[i].shape[0]
        cnrow.append(nrow)
    d2=[]
    for i in dict_list:
        T_inv=np.array(mu_dict[i].transpose())
        A_inv = np.linalg.inv(sigma_dict[i])
        nnrow=[0 if cnrow[i-1]==0 else cnrow[i-1]]
        d2.append(nnrow[0]*(np.matmul(np.matmul(T_inv,A_inv),np.array(mu_dict[i]))))
    d2=round(sum(d2),2)
    df=sum(list(kj_dict.values()))-n_var
    p_value=round(1-stats.chi2.cdf(d2,df),3)
    # interpretation
    if p_value >= alpha:
        con='completely random'
    if p_value < alpha:
        con='not random'
    restab=pd.DataFrame({'statistic':[d2],'df':[df],'p.value':[p_value],'missing.patterns':[n_miss_pattern],'missing.data':[con]})
    return restab


def object_miss(df,nORp=0.2):
    """
    For obtain the number of the rows in a data frame that have a "large" number of missing values. "Large" can be defined either as a proportion of the number of columns or as the number in itself.

    Parameters
    ----------
    df : [DataFrame] dataframe to be verified for missing values.
    nORp : [float] a number controlling when a row is considered to have too many NaN values (defaults nORp=0.2, i.e. 20% of the columns).

    Returns
    ----------
    Count of objects with missing values and a list with the IDs of the rows with too many NA values
    """
    data=df.T
    notnull=data.notnull().sum()
    null=data.isnull().sum()
    missdf=pd.DataFrame({'exist':null,'missing':notnull,'ratio':null/notnull})
    for x in missdf.ratio:ind=missdf.index[missdf.ratio>nORp].tolist()
    print(f"{len(ind)} objects with missing values")
    return ind


def prelim_norm(x):
    """
    Perform preliminary manipulations on matrix of continuous data.
    Rows are sorted by missing data pattern.

    Parameters
    ----------
    x : [DataFrame] data frame containing missing values. The rows of x correspond 
                    to observational units, and the columns to variables. Missing values are denoted by NA.

    Returns
    ----------
    a list of thirteen components that summarize various features of x after the data have been centered,
    scaled, and sorted by missingness patterns. Components that might be of interest to the user include:
    nmis : a vector of length x.shape[0] containing the number of missing values for each variable in x.
           This vector has names that correspond to the column names of x, if any.
    r :    matrix of response indicators showing the missing data patterns in x. Dimension is (S,p) where
           S is the number of distinct missingness patterns in the rows of x, and p is the number 
           of columns in x. Observed values are indicated by 1 and missing values by 0. The row names 
           give the number of observations in each pattern, and the column names correspond to the 
           column names of x.
    """
    def uniqueIndexes(l):
        """
        Obtaining indexes of unique values in the list.
        """
        l=list(l.sort_values())
        seen = set()
        res = []
        for i, n in enumerate(l):
            if n not in seen:
                res.append(i)
                seen.add(n)
        return res
    def mkpsi(p):
        """
        Generates a symmetric matrix of integers indicating the linear position in packed 
        storage of the matrix elements.
        """
        d = np.full((p+1,p+1), 0)
        d[:, 0] = [i for i in range(1,p+2)]
        d[0]=[i for i in range(1,p+2)]
        l=[i for i in reversed(range(1,p+2))]
        for i in range(1,p+1):
            for k in range(1,p+1):
                if i == k:
                    d[i,k] = d[i-1,k-1]+l[i-1]
        for i in range(1,p+1):
            for k in range(1,p+1):
                if i < k:
                    d[i,k] = d[i-1,k-1]+l[i-1]
        for i in range(2,p+1):
            for k in range(1,p):
                if i > k:
                    d[i,k] = d[i-1,k-1]+l[k-1]
        return d
    def sjn(p,npatt,r):
        """
        Computes s_j, the number of the last missingness pattern for which the jth 
        variable needs to be imputed to complete the monotone pattern.
        """
        r['count']=range(r.shape[0])
        r=r.set_index('count')
        r = r.rename_axis(index=None)
        sj=[0]*p
        for i in range(p):
            patt=npatt
            if r[r.columns[i]][npatt-1]!=0:
                sj[i]=patt
            if r[r.columns[i]][npatt-1]==0:
                patt=patt-1
                sj[i]=patt
        tmp=sj[p-1]
        for i in range(p-2, -1, -1):
            pt=np.max([sj[i],tmp])
            sj[i]=pt
        return sj
    def nmons(p,sj,nmdp):
        """
        Computes the number of observations in (Xobs,Xmis*) for each column.
        """
        nmon=[0]*p
        for i in range(p):
            nmon[i]=0
            nmon1=[]
            for patt in range(sj[i]):
                nm=nmon[i]+nmdp[patt]
                nmon1.append(nm)
                sumnm=sum(nmon1)
            nmon[i]=sumnm
        return nmon
    def lasts(p,npatt,sj):
        """
        Finds last variable in each missingness pattern to complete a monotone pattern.
        """
        last=[0]*(npatt+1)
        for i in range(p,-1,-1):
            if i==p:
                start=0
            else:
                start=sj[i]
            lasts=[]
            for patt in range(start,sj[i-1]):
                last[patt]=i
        last.pop(npatt)
        return last
    def layers(p,sj):
        """
        Finds layers for observed parts of the sufficient statistics.
        """
        nlayer=0
        layer=[0]*(p+1)
        for i in range(p-1,-1,-1):
            if i+1 == p:
                if sj[i]>0:
                    nlayer=nlayer+1
            else:
                if sj[i]>sj[i+1]:
                    nlayer=nlayer+1
            layer[i+1]=nlayer
        layer.pop(0)
        return layer
    # get dimensions of x
    if isinstance(x, pd.DataFrame):
        x=x.copy()
    else:
        x=pd.DataFrame(x)
    p=x.shape[1]
    n=x.shape[0]
    var_names=list(x.columns)
    # find missingness patterns
    r=pd.DataFrame(data=np.where(x.isna(),1,0),columns=var_names)
    nmis=pd.DataFrame(x.isnull().sum()).T
    # index the missing data patterns
    mdp=np.matmul(r,[2 ** num for num in list(range(0,p))])+1
    # do row sort
    ro=[i[0] for i in sorted(enumerate(mdp), key=lambda x:x[1]+1)]
    x=x.loc[x.index[ro]]
    mdp=list(mdp.sort_values())
    r=r.loc[r.index[ro]]
    ro=[i[0] for i in sorted(enumerate(ro), key=lambda x:x[1]+1)]
    # compress missing data patterns
    mdpst=uniqueIndexes(pd.Series(mdp))
    mdp=np.sort(pd.Series(mdp).unique())
    npatt=len(mdpst)
    # create r-matrix for display purposes
    r1=pd.DataFrame(data=np.where(r==0,1,0),columns=var_names)
    r2=r1.loc[r1.index[mdpst]].reset_index()
    r2 = r2.drop(r2.columns[[0]], axis=1)
    cl=[]
    for i in range(0,r2.shape[0]):
        df3 = r1.merge(pd.DataFrame(r2.loc[i:i]), on=list(r1.columns))
        count=df3.shape[0]
        cl.append(count)
    r2['count']=cl
    r2 = r2.set_index('count')
    r2 = r2.rename_axis(index=None)
    # center and scale the columns of x
    if x.isnull().sum().sum()<x.shape[1]:
        mvcode=x.max().max()+1000
        x=x.fillna(mvcode)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(x.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=[x.columns.to_list()])
    xbar=list(x.mean().values.round(6))
    sdv=list(x.std(ddof=0).values.round(6))
    if x.isnull().sum().sum()==x.shape[1]:
        xbar=[0] * p
        sdv=[1] * p
    # form matrix of packed storage indices
    d=int((2+3*p+p**2)/2)
    psi=mkpsi(p)
    # other bookkeeping quantities
    if npatt>1:
        nmdp=list(r2.index)
    if npatt==1:
        nmdp=n
    sj=sjn(p,npatt,r2)
    nmon=nmons(p,sj,nmdp)
    last=lasts(p,npatt,sj)
    layer=layers(p,sj)
    nlayer=np.max(layer)
    # return dictionary
    res={'x':df_scaled, 'n':n, 'p':p, 'r':r2, 'nmis':nmis, 'ro':ro, 'mdpst':mdpst,
        'nmdp':nmdp, 'npatt':npatt, 'xbar':xbar, 'sdv':sdv, 'd':d, 'psi':psi, 'sj':sj,
        'nmon':nmon, 'last':last, 'layer':layer, 'nlayer':nlayer}
    return res


def rangen(init):
    """
    Generates random numbers for a Gaussian distribution.
    """
    a=16807
    b15=32768
    b16=65536
    p=2147483647
    if init==0:
        ix=init
        rangen=float(ix)*4.656612875E-10
        return rangen
    else:
        if init>0:
            ix=init
            xhi=int(ix/b16)
            xalo=int((ix-xhi*b16)*a)
            leftflo=int(xalo/b16)
            fhi=int(xhi*a+leftflo)
            k=int(fhi/b15)
            ix=int((((xalo-leftflo*b16)-p)+(fhi-k*b15)*b16)+k)
            if ix<0:
                ix=ix+p
            rangen=float(ix)*4.656612875E-10
            return rangen
        if init<0:
            ix=init
            xhi=int(ix)/int(b16)
            xalo=int((ix-xhi*b16)*a)
            leftflo=int(xalo/b16)
            fhi=int(xhi*a+leftflo)
            k=int(fhi/b15)
            ix=int((((xalo-leftflo*b16)-p)+(fhi-k*b15)*b16)+k)
            rangen=float(ix)*4.656612875E-10
            return rangen

def sigex(d,theta,extr,p,psi,mc,nmc):
    """
    Extracts submatrix of theta corresponding to the columns of mc.
    """
    theta=theta.copy()
    extr=extr.copy()
    for j in range(1,nmc):
        for k in range(j,nmc):
            extr[psi[mc[j],mc[k]]]=theta[(psi[mc[j],mc[k]])-1]
    return extr


def swp(d,theta_,pivot,p,psi,submat,diR):
    """
    Performs sweep on a symmetric matrix in packed storage. Sweeps on pivot position.
    Sweeps only the (0:submat,0:submat) submatrix. If dir=1, performs ordinary sweep.
    If dir=-1, performs reverse sweep.
    """
    theta_=theta_.copy()
    a=theta_[psi[pivot,pivot]-1]
    theta_[psi[pivot,pivot]-1]=-1/a
    for i in (range(submat+1)):
        if i != pivot:
            theta_[psi[i,pivot]-1]=theta_[psi[i,pivot]-1]/a*diR
    for i in range(submat+1):
        for j in range(i,submat+1):
            if i != pivot and j != pivot:
                b=theta_[psi[i,pivot]-1]
                c=theta_[psi[j,pivot]-1]
                theta_[psi[i,j]-1]=theta_[psi[i,j]-1]-a*b*c
    return theta_


def swpobs(d,theta_1,p,psi,npatt,r,patt):
    """
    Sweeps theta to condition on the observed variables.
    """
    theta_1=theta_1.copy()
    for j in range(1,p+1):
        if r.iloc[patt,j-1]==1 and theta_1[psi[j,j]-1]>0:
            theta_1=swp(d,theta_1,j,p,psi,p,1)
        if r.iloc[patt,j-1]==0 and theta_1[psi[j,j]-1]<0:
            theta_1=swp(d,theta_1,j,p,psi,p,-1)
    return theta_1


def tobsn(d,p,psi,n,x,npatt,r,mdpst,nmdp):
    """
    Tabulates the known part of the sscp matrix for all missingness patterns.
    """
    tobs=initn(d)
    for patt in range(npatt):
        oc,noc=gtoc(p,npatt,r,patt)
        for i in range(mdpst[patt],mdpst[patt]+nmdp[patt]):
            for j in range(noc):
                tobs[(psi[0,oc[j]])-1]=tobs[(psi[0,oc[j]])-1]+x.iloc[i,oc[j]-1]
                for k in range(j,noc):
                    tobs[(psi[oc[j],oc[k]])-1]=tobs[(psi[oc[j],oc[k]])-1]+x.iloc[i,oc[j]-1]*x.iloc[i,oc[k]-1]
    return tobs