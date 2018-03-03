# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:56:49 2018

@author: Rajesh Rajendran
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
from sklearn.model_selection import train_test_split

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
upset_times1 = pd.to_datetime(['26-Oct-2016', '11-Nov-2016', '16-Nov-2016', 
                              '27-Nov-2016', '28-Jan-2017', '12-Mar-2017' ])


upset_times = pd.to_datetime(['26-Oct-2016 08:45:00 AM', '11-Nov-2016 08:00:00 AM', '16-Nov-2016 06:45:00 AM', 
                              '27-Nov-2016 06:00:00 AM', '28-Jan-2017 07:15:00 AM', '12-Mar-2017 07:00:00 AM' ])

rollback_times = [3, 6]
data1 = pd.read_table(file1, encoding='utf-8')
data1.columns = c[c['Col0'] == 1]['Col2']
ts = lambda x : pd.to_datetime(x)
data1['TS'] = data1.TimeStamp.apply(ts)

print(FeatureCols)
for u in upset_times:
    print(' Processing date : ' + str(u))
    r = [u + relativedelta(hours=23.99), u - relativedelta(hours=24)]
    data2 = data1[(data1.TS >= r[1]) & (data1.TS <= r[0])]
    data2['TimeStamp'] = data2.TimeStamp.apply(HourlyMatch)
    dttime = set(data2['TimeStamp'])
    data3 = pd.DataFrame()
    i = 0
    for d in dttime:
        i = i + 1
        #print(str(i) + ' Processing : ' + d)
        x = data2[data2['TS'] == d].describe()
        x.reset_index()
        x['timestamp'] = d
        data3= data3.append(x)
        
    del data3['PIIntShapeID']
    data3 = data3.reset_index()
    data3=data3[data3['index'] == 'mean']
    #plot_graph(data3, tag=FeatureCols, version=['mean'], failures=0)
    l  = len(data3['5PT1B2'])
    plt.plot(list(range(0,l)), data3[FeatureCols])
    plt.show()


times = []
for u1 in upset_times:
    z1 = u1 + relativedelta(hours=24)
    z2 = u1 - relativedelta(hours=24)
    while z2 <= z1:
        times.append(z2)
        z2 = z2 + relativedelta(hours=1)


final_data = pd.DataFrame()
range_value = 1 #hrs
for u1 in times:
    for r1 in rollback_times:
        u2 = u1 - relativedelta(hours = r1)
        print(u2)
        deltarange1 = u2 - relativedelta(hours=range_value)
        deltarange2 = u2 + relativedelta(hours=range_value)
        print(deltarange1)
        print(deltarange2)
        ds = data1[(data1.TS >= deltarange1) & (data1.TS <= deltarange2)]
        ds = ds.describe()
        ds = ds.reset_index()
        ds ['timestamp'] = u1
        ds ['rollback_hrs'] = r1
        print(len(ds))
        final_data = final_data.append(ds)


def UpsetTimes1(dt):
    flag = 0
    
    for d in upset_times:
        #print(d)
        if dt == d:
            flag = 1

    return flag

final_data['upsets'] = final_data['timestamp'].apply(UpsetTimes1)
mean3 = final_data[(final_data.rollback_hrs == 3) & (final_data['index'] == 'mean')]
mean6 = final_data[(final_data.rollback_hrs == 6) & (final_data['index'] == 'mean')]

l  = len(mean3)
plt.plot(list(range(0,l)), mean3[FeatureCols])
plt.show()
l  = len(mean6)
plt.plot(list(range(0,l)), mean6[FeatureCols])
plt.show()


#LogisticRegression Model
def ModelLogistics(df, version='mean', rollback=3):
    C1 = 10000
    miter = 1000
    
    meanvalue = final_data[(final_data.rollback_hrs == rollback) & (final_data['index'] == version)]

    dftagdata = meanvalue[['5PT1B2', '5PT3B2', '5PT2C1', '5PT3C1', '5PT2G1', '5PT3G1', '5PT1H1', '5PT4H1', '5TT1B2', '5TT3B2', '5TT2C1', '5TT3C1', '5TT2G1', '5TT3G1', '5TT1H1', '5TT4H1']]
    #dftagdata = meanvalue[FeatureCols]
    dffaildata = meanvalue [faildatacols]
    x_train, x_test, y_train, y_test = train_test_split(dftagdata, dffaildata, random_state=0)
    
    clf = DecisionTreeClassifier(random_state = 0).fit(x_train, y_train)
    FeatureCols = pd.DataFrame(clf.feature_importances_, x_train.columns.tolist())
    FeatureCols = FeatureCols.sort_values(by = 0, ascending = False)
    print(FeatureCols)
    FeatureCols = list(FeatureCols[FeatureCols[0] >= 0.05].index)

    dfdata = meanvalue 
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
    print(y_train.reshape(1,-1))
    print(pln.predict(x_test))
    print(y_test.reshape(1,-1))
    
 
    
ModelLogistics(final_data,rollback=3,version='mean' )
min Train Score - 0.990909090909091 Test Score - 0.9594594594594594

ModelLogistics(final_data,rollback=6,version='mean')
min Train Score - 0.9954545454545455 Test Score - 0.9324324324324325

ModelLogistics(final_data,version='std', rollback=3)
min Train Score - 0.9863636363636363 Test Score - 0.9594594594594594

ModelLogistics(final_data,version='std', rollback=6)
min Train Score - 0.990909090909091 Test Score - 0.9459459459459459

