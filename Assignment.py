# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 07:04:24 2018

@author: Rajesh Rajendran
Description - The below code is developed for a Hackathon 
organzied by BHP to predict the perdict the upset times for 
the 3 phase separators cased on the sensors data.
"""
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta
from itertools import repeat
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

#Common functions or variables
#Preparation for features generation
tagdatacols = ['5PT1B2', '5PT3B2', '5PT2C1', '5PT3C1', '5PT2G1', '5PT3G1', '5PT1H1', 
               '5PT4H1', '5TT1B2', '5TT3B2', '5TT2C1', '5TT3C1', '5TT2G1', '5TT3G1', 
               '5TT1H1', '5TT4H1', '20PICP2Choke', '20PICP1Choke', '20PT17FLDC', 
               '20PT18FLDC', '20PT27FLDC', '20PT28FLDC', '20PT214FLLA','20PT224FLRE', 
               '20TT115FLT1', '20TT125FLTS', '20TT215T2FLL', '20TT225T2FLL', 
               '20ZT114SSFL', '20ZT124SSFL', '20ZT214T2FL', '20ZT224T2FL', '21FQI10518NR', 
               '21FT40518D', '21FT40518GVFR', '21HY10535OFL', '21HY40534OTSL', 
               '21LIC10516SP', '21LIC10620CVH', '21LIC10620SPH', '21LIC40516SPTA', 
               '21LT10515PVPSO', '21LT10516PVPSO', '21LT10618PVPSO2', '21LT10620PVPSO2', 
               '21LT40515PVTA', '21LT40516PVTA', '21LY10516OPSO2', '21LY10616OPSO2', 
               '21LY10620OSH2', '21LY11516OTT', '21LY40516OUT', '21PT10505PVPS', 
               '21PT10605PVPS2', '21PT40505PVTA', '21TT10508PVPSO', '21TT10608PVPSO2', 
               '21TT11616PVOTHO', '30FT19107PVSH2', '30FT19108PV', '30FT29108PV', 
               '30FT69521PVFCP', '30LIC69516CVFCO', '30LIC69518CVFCP', '30LT69514PVFC', 
               '30LT69515PVFC', '30LT69516PVFC', '30LT69518PVFC', '30LY69518OFCP', 
               '30PDIC19104SPPHO', '30PDT19104PVSH2', '30PDT19104PVSHS2', 
               '30PDT19104PVSHD2', '30PDY19104OSPH2', '30PT69503PVFC', '30PT69512PV', 
               '30PY69503OFCO', '37PT62301PVCS']
SubseaTags = ['5PT1B2', '5PT3B2', '5PT2C1', '5PT3C1', '5PT2G1', '5PT3G1', '5PT1H1', 
              '5PT4H1', '5TT1B2', '5TT3B2', '5TT2C1', '5TT3C1', '5TT2G1', '5TT3G1', 
              '5TT1H1', '5TT4H1']
FlowlineTags = ['20PT17FLDC', '20PT18FLDC', '20PT27FLDC', '20PT28FLDC',	'20TT115FLT1', 
                '20TT125FLTS'	, '20TT215T2FLL',	 '20TT225T2FLL', '20ZT114SSFL', 
                '20ZT124SSFL',	 '20ZT214T2FL'	, '20ZT224T2FL'	]
EqpsensorTags = ['21FQI10518NR', '21FT40518D', '21FT40518GVFR', '21HY10535OFL', '21HY40534OTSL', '21LIC10516SP',
                 '21LIC10620CVH', '21LIC10620SPH',  '21LIC40516SPTA', '21LT10515PVPSO', '21LT10516PVPSO', '21LT10618PVPSO2',
                 '21LT10620PVPSO2', '21LT40515PVTA',  '21LT40516PVTA', '21LY10516OPSO2', '21LY10616OPSO2', '21LY10620OSH2', 
                 '21LY11516OTT',  '21LY40516OUT', '21PT10505PVPS', '21PT10605PVPS2', '21PT40505PVTA', '21TT10508PVPSO', 
                 '21TT10608PVPSO2', '21TT11616PVOTHO', '30FT19107PVSH2', '30FT19108PV', '30FT29108PV', '30FT69521PVFCP', 
                 '30LIC69516CVFCO', '30LIC69518CVFCP', '30LT69514PVFC', '30LT69515PVFC', '30LT69516PVFC', '30LT69518PVFC',  
                 '30LY69518OFCP', '30PDIC19104SPPHO', '30PDT19104PVSH2', '30PDT19104PVSHS2', '30PDT19104PVSHD2', 
                 '30PDY19104OSPH2', '30PT69503PVFC', '30PT69512PV', '30PY69503OFCO', '37PT62301PVCS']
faildatacols = ['upsets']
summaryfields = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
upset_times = pd.to_datetime(['26-Oct-2016', '11-Nov-2016', '16-Nov-2016', 
                              '27-Nov-2016', '28-Jan-2017', '12-Mar-2017' ])

def CreateTicks(m1, m2, step=10):
    ticks = []
    t = m1
    while t <= m2:
        ticks.append (t)
        t = t +  relativedelta(days=step)
    
    return ticks

def HourlyMatch(s):
    d = str.split(s, ':')[0]+':00:00'
    if s.find('AM') > 0:
        d = d + ' ' + 'AM'
    else:
        d = d + ' ' + 'PM'
    return d

def UpsetTimes(dt):
    flag = 0
    
    for d in upset_times:
        #print(d)
        if dt.date() == d.date():
            flag = 1

    return flag

#Read data from files
file1 = 'Hackathon_DataSet_OctApr_Part1.txt'
file2 = 'Hackathon_DataSet_OctApr_Part2.txt'
cols = 'Columns.txt' 

c = pd.read_table(cols, encoding='utf-8')

df1 = pd.read_table(file1, encoding='utf-8')
df1.columns = c[c['Col0'] == 1]['Col2']
df2 = pd.read_table(file2, encoding='utf-8')
df2.columns = c[c['Col0'] == 2]['Col2']

for s in df2.columns:
    if (s != 'Id' and s != 'hackathon4' and s != 'TimeStamp' and s != 'PIIntTSTicks' and s != 'PIIntShapeID'):
        df1[s] = df2[s]

del df2
df2 = df1

#Initial data preparations
df1['TS'] = df1.TimeStamp.apply(HourlyMatch)

dttime = set(df1['TS'])

df3 = pd.DataFrame()
i = 0
for d in dttime:
    i = i + 1
    print(str(i) + ' Processing : ' + d)
    x = df1[df1['TS'] == d].describe()
    x.reset_index()
    x['timestamp'] = d
    df3 = df3.append(x)

df3 = df3.reset_index()
df3['timestamp'] = df3['timestamp'].apply(pd.to_datetime)
df3 = df3.sort_values(by=['timestamp'])

#Place for removing the outliers. Follow either 1 sigma or 2 sigma model. Retain mean for outlier placeholders. 

#Generic Plotting to visualize data
def plot_graph(df, tag = ['5PT1B2'], version=['all'], failures=1):
    if version[0] == 'all':
        version = summaryfields 
    #print(version)
    
    for tg in tag:
        #print(tg)
        x = pd.DataFrame()
        x = df[['timestamp','index',tg]]

        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111)
        ax.text(.5,.9,'Tag - ' + list(c[c.Col2 == tg].Col1)[0], horizontalalignment='center', transform=ax.transAxes)
        plt.grid(b=True, which='both', color='0.65', linestyle='-')

        for v in version:
            #print(v)
            x1 = x[x['index'] == v]
            x1 = x1.reset_index()
            ticks = CreateTicks(x1['timestamp'][0], x1['timestamp'][len(x1)-1], step=10)
            
            ut = []
            for u in upset_times:
                ut.append(x1[x1['timestamp']==u].index[0])
            
            plt.xticks(range(0, len(x1), int(len(x1)/len(ticks))), ticks, rotation = '90')
            plt.plot(x1[tg], label=v)
        
        if failures==1:
            plt.scatter(ut, list(repeat(min(x1[tg]), len(ut))), label='Upsets', s=80, color='red', marker='o')
            
        plt.legend(loc = 2, frameon = True)
        plt.show()
    return

#Data Preparation for prediction Models
df2.TimeStamp = pd.to_datetime(df2.TimeStamp)
stephrs = 24
timesteps = [] 
ts = m1 = pd.to_datetime(list(df2['TimeStamp'])[0])
m2 = list(df2['TimeStamp'])[len(list(df2['TimeStamp']))-1]
while ts < m2:
    timesteps.append(ts)
    ts = ts + relativedelta(hours=stephrs)

df = pd.DataFrame()
for i in range(1, len(timesteps)-1):
    #print(timesteps[i], timesteps[i+1])
    print('Processing Range :' + str(timesteps[i]) + ', ' + str(timesteps[i+1]))
    x = df2[(df2.TimeStamp >= timesteps[i]) & (df2.TimeStamp >= timesteps[i+1])]
    x = x.describe()
    x['timestamp'] = timesteps[i+1]
    df = df.append(x)
    del x
    
df = df.reset_index()
df['upsets'] = df['timestamp'].apply(UpsetTimes)

#Decision Classifier - What columns or tags are important
#idx = 'std'
tagdata = SubseaTags
dfdata = df[tagdata]
dfdata['timestamp'] = df['timestamp']
dfdata['upsets'] = df['upsets']

desc = set(['std', 'min', 'max'])
for idx in desc:
    dfdata = df[df['index'] == idx]
    dftagdata = dfdata[tagdata]
    dffaildata = dfdata [faildatacols]
    x_train, x_test, y_train, y_test = train_test_split(dftagdata, dffaildata, random_state=0)

    clf = DecisionTreeClassifier(random_state = 0).fit(x_train, y_train)
    FeatureCols = pd.DataFrame(clf.feature_importances_, x_train.columns.tolist())
    FeatureCols = FeatureCols.sort_values(by = 0, ascending = False)
    FeatureCols = list(FeatureCols[FeatureCols[0] >= 0.05].index)
    print(FeatureCols)

#SVC Model
C1 = 10000
desc = set(['std', 'min', 'max'])
for idx in list(desc):
    print('SVC Modeling')
    dfdata = df[df['index'] == idx]
    dftagdata = dfdata[FeatureCols]
    dffaildata = dfdata [faildatacols]
    scaler = StandardScaler().fit(dftagdata)
    dftagdata = scaler.transform(dftagdata)
    
    x_train, x_test, y_train, y_test = train_test_split(dftagdata, dffaildata, random_state=0)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    pln = SVC(C=C1).fit(x_train, y_train)
    print (idx + ' Train Score - ' + str(pln.score(x_train, y_train)) + ' Test Score - ' + str(pln.score(x_test, y_test)))
    print(pln.predict(x_train))
    print(y_train.reshape(1,-1)[0])
    print(pln.predict(x_test))
    print(y_test.reshape(1,-1)[0])
    
#LogisticRegression Model
C1 = 10000
miter = 1000
desc = set(['std', 'min', 'max'])
for idx in list(desc):
    print('LogisticRegression Modeling')
    dfdata = df[df['index'] == idx]
    dftagdata = dfdata[FeatureCols]
    dffaildata = dfdata [faildatacols]
    scaler = StandardScaler().fit(dftagdata)
    dftagdata = scaler.transform(dftagdata)
    
    x_train, x_test, y_train, y_test = train_test_split(dftagdata, dffaildata, random_state=0)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    pln = LogisticRegression(C=C1, max_iter=miter).fit(x_train, y_train)
    print (idx + ' Train Score - ' + str(pln.score(x_train, y_train)) + ' Test Score - ' + str(pln.score(x_test, y_test)))
    print(pln.predict(x_train))
    print(y_train.reshape(1,-1)[0])
    print(pln.predict(x_test))
    print(y_test.reshape(1,-1)[0])
    
#LassoRegression Model
a = 0.1
miter = 1000
desc = set(['std', 'min', 'max'])
for idx in list(desc):
    print('Lasso Modeling')
    dfdata = df[df['index'] == idx]
    dftagdata = dfdata[FeatureCols]
    dffaildata = dfdata [faildatacols]
    scaler = StandardScaler().fit(dftagdata)
    dftagdata = scaler.transform(dftagdata)
    
    x_train, x_test, y_train, y_test = train_test_split(dftagdata, dffaildata, random_state=0)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    pln = Lasso(alpha=a, max_iter=miter).fit(x_train, y_train)    
    print (idx + ' Train Score - ' + str(pln.score(x_train, y_train)) + ' Test Score - ' + str(pln.score(x_test, y_test)))
    print(pln.predict(x_train))
    print(y_train.reshape(1,-1)[0])
    print(pln.predict(x_test))
    print(y_test.reshape(1,-1)[0])
    
#Polynomial Features addition
C1 = 10000
miter = 1000
deg = list(range(1,5))
desc = set(['std', 'min', 'max'])
print('Polynomial (degree from 1 to 5); LogisticRegression Modeling')
from sklearn.preprocessing import PolynomialFeatures
desc = set(['std', 'min', 'max'])
for idx in list(desc):
    dfdata = df[df['index'] == idx]
    dftagdata = dfdata[FeatureCols]
    dffaildata = dfdata [faildatacols]
    scaler = StandardScaler().fit(dftagdata)
    dftagdata = scaler.transform(dftagdata)
    
    x_train, x_test, y_train, y_test = train_test_split(dftagdata, dffaildata, random_state=0)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    for d in deg:
        print('Degree - ' + str(d))
        poly = PolynomialFeatures(degree=d)
        x_train_poly = poly.fit_transform(x_train)
        x_test_poly = poly.transform(x_test)
        pln = LogisticRegression(C=C1, max_iter=miter).fit(x_train_poly, y_train)
        print (idx + ' Train Score - ' + str(pln.score(x_train_poly, y_train)) + ' Test Score - ' + str(pln.score(x_test_poly, y_test)))
        print(pln.predict(x_train_poly))
        print(y_train.reshape(1,-1)[0])
        print(pln.predict(x_test_poly))
        print(y_test.reshape(1,-1)[0])
        
plot_graph(df, tag = FeatureCols, version = ['std'])

#plot_graph(df3, tag = ['5PT1B2', '5PT3B2', '5TT1B2'],version=['max', 'min'])
#plot_graph(df3, tag = ['5PT1B2', '5PT3B2', '5PT2C1', '5PT3C1', '5PT2G1','5PT3G1', '5PT1H1', '5PT4H1', '5TT1B2', '5TT3B2', '5TT2C1', '5TT3C1','5TT2G1', '5TT3G1', '5TT1H1', '5TT4H1'],version=['max', 'min'])
#plot_graph(df3, tag = ['20PICP2Choke', '20PICP1Choke','20PT17FLDC', '20PT18FLDC', '20PT27FLDC', '20PT28FLDC', '20PT214FLLA', '20PT224FLRE', '20TT115FLT1', '20TT125FLTS', '20TT215T2FLL', '20TT225T2FLL', '20ZT114SSFL', '20ZT124SSFL', '20ZT214T2FL', '20ZT224T2FL', '21FQI10518NR', '21FT40518D', '21FT40518GVFR', '21HY10535OFL', '21HY40534OTSL', '21LIC10516SP', '21LIC10620CVH', '21LIC10620SPH', '21LIC40516SPTA', '21LT10515PVPSO', '21LT10516PVPSO', '21LT10618PVPSO2', '21LT10620PVPSO2', '21LT40515PVTA', '21LT40516PVTA', '21LY10516OPSO2', '21LY10616OPSO2', '21LY10620OSH2', '21LY11516OTT', '21LY40516OUT', '21PT10505PVPS', '21PT10605PVPS2', '21PT40505PVTA', '21TT10508PVPSO', '21TT10608PVPSO2', '21TT11616PVOTHO', '30FT19107PVSH2', '30FT19108PV', '30FT29108PV', '30FT69521PVFCP', '30LIC69516CVFCO', '30LIC69518CVFCP', '30LT69514PVFC', '30LT69515PVFC', '30LT69516PVFC', '30LT69518PVFC', '30LY69518OFCP', '30PDIC19104SPPHO', '30PDT19104PVSH2', '30PDT19104PVSHS2', '30PDT19104PVSHD2', '30PDY19104OSPH2', '30PT69503PVFC', '30PT69512PV', '30PY69503OFCO', '37PT62301PVCS' ],version=['max', 'min'])
#
#def YearMonth(t):
#    return pd.to_datetime(str(t.year) + '-' + str(t.month) + '-01') 
#
#def YearMonthList(s):
#    return str(s.month) + '-' + str(s.year)
#
'''
string_tags = ['20HX13FL','20HX14FL','20HX23FL','20HX24FL']
   
df = df1[df1['TS'] == '3/31/2017 11:00:00 PM']
df = df.append(df1[df1['TS'] == '10/20/2016 12:00:00 AM'])
df.describe()

string columns '20HX13FL','20HX14FL','20HX23FL','20HX24FL'
file = 'data.xlsx'
writer = pd.ExcelWriter(file)
piv = df1[['Id', 'hackathon4', 'TimeStamp','PIIntTSTicks', 'PIIntShapeID']]
piv.to_excel(writer, sheet_name='Sheet1')
piv = df2[['Id', 'hackathon4', 'TimeStamp','PIIntTSTicks', 'PIIntShapeID']]
piv.to_excel(writer, sheet_name='Sheet2')
writer.save()

fig = plt.figure(figsize=(15,10))
plt.plot(df[0],df[1], 'r')
'''
trainscore = pd.DataFrame()
trainscore['x_train'] = pln.predict(x_train)
trainscore['y_train'] = y_train.reshape(1,-1)[0]
trainscore[(trainscore.x_train == 1) | (trainscore.y_train == 1)]
testscore  = pd.DataFrame()
testscore['x_test'] = pln.predict(x_test)
testscore['y_test'] = y_test.reshape(1,-1)[0]
testscore[(testscore.x_test == 1) | (testscore.y_test == 1)]

from scipy.stats.stats import pearsonr
pearsonr(df1['5PT4H1'], df1['5PT2C1'])

#From data perspective it is an imbalanced class; Random subsampling of majority class. 
#Synthetic Minority oversampling technique ; SMOT
#Irregular time series pattern. 
#Not all variances results in upset.
#Inconsitant timelag due to steep variances in flowrate. 
#Upsets -> Flowrates -> subsea sensors... They are the features for our analysis. 
