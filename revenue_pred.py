import pandas as pd
import numpy as np
import matplotlib. pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
df = pd.read_csv("Revenue.csv")
# print(df)
# print(df.columns)
#type - fc - food court, IL - inline , DT- drive, MB - mobile

# print(df['Open Date'].unique())
df['Open Date'] = pd.to_datetime(df['Open Date'],format="%m/%d/%Y")
df['OpenDays'] = ''
datelast = pd.DataFrame({'Date':np.repeat(['01/01/2018'],len(df))})
datelast['Date'] = pd.to_datetime(datelast['Date'],format="%m/%d/%Y")
df['OpenDays'] = datelast['Date'] - df['Open Date']
df['OpenDays'] = df['OpenDays'].astype('timedelta64[ns]').astype('int64')
# print(df.head())
df = df.drop('Open Date', axis = 1)

city_perc = df[['City Group','revenue']].groupby(['City Group'],as_index=False).mean()

sns.barplot(x = 'City Group', y = 'revenue', data = city_perc)
plt.show()

city_perc = df[['City','revenue']].groupby(['City'],as_index=False).mean()
newdf = city_perc.sort_values(['revenue'],ascending = True)
sns.barplot(x = 'City', y = 'revenue', data = newdf.head(10))
plt.show()

city_perc = df[['Type','revenue']].groupby(['Type'],as_index=False).mean()
sns.barplot(x = 'Type', y = 'revenue', data = city_perc)
plt.show()

city_perc = df[['Type','OpenDays']].groupby(['Type'],as_index=False).mean()
sns.barplot(x = 'Type', y = 'OpenDays', data = city_perc)
plt.show()

df = df.drop('Id',axis = 1)
df = df.drop('Type',axis = 1)

citygroup_dummies = pd.get_dummies(df['City Group'])
df = df.join(citygroup_dummies)

df = df.drop('City Group',axis = 1)
df = df.drop('City',axis = 1)
# print(df.head())
# print(df.columns)
y = df[['revenue']]
x = df.drop(columns=['revenue'])
# print(x.columns)

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
print(x_train.shape ,x_test.shape,y_train.shape,y_test.shape )

from sklearn.ensemble import RandomForestClassifier

fea_lables = x.columns
forest = RandomForestClassifier(n_estimators=500 , random_state=1)
forest.fit(x_train,y_train)
importance = forest.feature_importances_
indicies = np.argsort(importance)[::-1]

for  i in range(x_train.shape[1]):
    print(fea_lables[indicies[i]], importance[indicies[i]])

best_features = df[fea_lables[indicies[0:19]]]
print(best_features.columns)
y = df[['revenue']]
x = best_features
# # print(x.columns)

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
print(x_train.shape ,x_test.shape,y_train.shape,y_test.shape )

from  sklearn.ensemble import RandomForestRegressor

cls = RandomForestRegressor(n_estimators=250,criterion='friedman_mse',max_depth=30)
cls.fit(x_train,y_train)
model_score = cls.score(x_train,y_train)
print(model_score)

pred = cls.predict(x_test)
r= []
for z in zip(y_test,pred):
    r.append(z)

    # print(z, (z[0]-z[1])/z[0])

plt.plot(r)