#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
import requests
import json


# In[2]:


# Setting this option will print all collumns of a dataframe
pd.set_option('display.max_columns',  None)
# Setting this option will print all of the data in a feature
pd.set_option('display.max_colwidth', None)


# In[3]:


def getBoosterVersion(data):
    for x in data['rocket']:
        if x:
            response= requests.get("https://api.spacexdata.com/v4/rockets/"+str(x)).json()
            BoosterVersion.append(response['name'])


# In[4]:


def getLaunchSite(data):
    for x in data['launchpad']:
        if x:
            response = requests.get("https://api.spacexdata.com/v4/launchpads/"+str(x)).json()
            Longitude.append(response['longitude'])
            Latitude.append(response['latitude'])
            LaunchSite.append(response['name'])
            


# In[5]:


def getPayloadData(data):
    for load in data['payloads']:
       if load:
        response = requests.get("https://api.spacexdata.com/v4/payloads/"+load).json()
        PayloadMass.append(response['mass_kg'])
        Orbit.append(response['orbit'])


# In[6]:


def getCoreData(data):
    for core in data['cores']:
            if core['core'] != None:
                response = requests.get("https://api.spacexdata.com/v4/cores/"+core['core']).json()
                Block.append(response['block'])
                ReusedCount.append(response['reuse_count'])
                Serial.append(response['serial'])
            else:
                Block.append(None)
                ReusedCount.append(None)
                Serial.append(None)
            Outcome.append(str(core['landing_success'])+' '+str(core['landing_type']))
            Flights.append(core['flight'])
            GridFins.append(core['gridfins'])
            Reused.append(core['reused'])
            Legs.append(core['legs'])
            LandingPad.append(core['landpad'])


# In[7]:


spacex_url="https://api.spacexdata.com/v4/launches/past"


# In[8]:


response = requests.get(spacex_url)


# In[9]:


print(response.content)


# In[10]:


static_json_url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'


# In[11]:


response.status_code


# In[12]:


data = response.json()


# In[13]:


data = pd.json_normalize(data)


# In[14]:


data.head()


# In[15]:


data['rocket']


# In[16]:


data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]


# In[17]:


data = data[data['cores'].map(len)==1]
data = data[data['payloads'].map(len)==1]


# In[18]:


data['cores'] = data['cores'].map(lambda x : x[0])
data['payloads'] = data['payloads'].map(lambda x : x[0])


# In[19]:


data['date'] = pd.to_datetime(data['date_utc']).dt.date


# In[20]:


data = data[data['date'] <= datetime.date(2020, 11, 13)]


# In[21]:


BoosterVersion = []
PayloadMass = []
Orbit = []
LaunchSite = []
Outcome = []
Flights = []
GridFins = []
Reused = []
Legs = []
LandingPad = []
Block = []
ReusedCount = []
Serial = []
Longitude = []
Latitude = []


# In[22]:


BoosterVersion


# In[23]:


getBoosterVersion(data)


# In[24]:


getLaunchSite(data)


# In[25]:


getPayloadData(data)


# In[26]:


getCoreData(data)


# In[27]:


launch_dict = {'FlightNumber': list(data['flight_number']),
'Date': list(data['date']),
'BoosterVersion':BoosterVersion,
'PayloadMass':PayloadMass,
'Orbit':Orbit,
'LaunchSite':LaunchSite,
'Outcome':Outcome,
'Flights':Flights,
'GridFins':GridFins,
'Reused':Reused,
'Legs':Legs,
'LandingPad':LandingPad,
'Block':Block,
'ReusedCount':ReusedCount,
'Serial':Serial,
'Longitude': Longitude,
'Latitude': Latitude}


# In[28]:


data_falcon9 = pd.DataFrame.from_dict(launch_dict)


# In[29]:


data_falcon9.shape


# In[30]:


data_falcon9.reset_index(inplace=True)
data_falcon9.head()


# In[31]:


data_falcon9.loc[:,'FlightNumber'] = list(range(1, data_falcon9.shape[0]+1))
data_falcon9


# In[32]:


df9 = data_falcon9


# In[33]:


df9.isnull().sum()


# In[34]:


mean = df9['PayloadMass'].mean()


# In[35]:


df9['PayloadMass'].replace(np.nan, mean, inplace=True)


# In[36]:


df9.head()


# In[37]:


df9[3:]
df9.head()


# In[38]:


data_falcon9.to_csv('datasetpart1.csv', index=False)


# In[39]:


df9 = pd.read_csv('datasetpart1.csv')


# In[40]:


df9.isnull().sum()/df9.shape[0]*100


# In[41]:


df9['Orbit'].value_counts()


# In[42]:


df9['LaunchSite'].value_counts()


# In[43]:


landing_outcomes = df9['Outcome'].value_counts()


# In[44]:


for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)


# In[45]:


bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
bad_outcomes


# In[46]:


f1=[]
for i in df9['Outcome']:
    if i in bad_outcomes:
        f1.append(0)
    else:
        f1.append(1)
        


# In[47]:


df9['Class'] = f1


# In[48]:


df9.head()


# In[49]:


df9["Class"].tail()
df9['Date'].astype


# In[50]:


landing_outcomes.isna().sum()


# In[51]:


df9 = df9[4::]


# In[52]:


get_ipython().run_line_magic('sql', 'select count(\'Mission_Outcome\') as MOC, Launch_Site from df9 group by "Launch_Site";')


# In[ ]:


import seaborn as sns
sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df9, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show()


# In[ ]:


sns.catplot(x='FlightNumber', y='LaunchSite', hue = "Class", data = df9)


# In[ ]:


orbit_success = df9.groupby('Orbit').mean()
orbit_success.reset_index(inplace=True)


# In[ ]:


sns.barplot(x = 'Orbit', y ='Class', data=orbit_success)


# In[ ]:


sns.catplot(x='FlightNumber', y='Orbit', hue = "Class", data = df9)


# In[ ]:


sns.catplot(x='PayloadMass', y='Orbit', hue = "Class", data = df9)


# In[ ]:


df9.head()


# In[ ]:



year = []
def Extract_year():
    for i in df9['Date'].astype('str'):
        year.append(i.split("-")[0])
    return pd.DataFrame(year)
date = Extract_year()


# In[ ]:


df9['Date'] = year
success_rate_per_y = df9.groupby('Date').mean()
success_rate_per_y.reset_index(inplace=True)


# In[ ]:


plt.plot(success_rate_per_y['Date'], success_rate_per_y['Class'])


# In[ ]:


features = df9[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()


# In[ ]:


features_one_hot = pd.get_dummies(features, columns=['Orbit','LaunchSite','LandingPad', 'Serial'])


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


ohe = OneHotEncoder()


# In[ ]:


ohe.fit(features_one_hot)


# In[ ]:


features_one_hot = features_one_hot.astype('float64')


# In[ ]:


features_one_hot.to_csv('dataset_part_3.csv', index=False)


# In[ ]:


features_one_hot


# In[ ]:


df_lat = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv")


# In[ ]:


spacex_df = df_lat[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
launch_sites_df


# In[ ]:


import folium
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon


# In[ ]:


nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=10)


# In[ ]:


circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
marker = folium.map.Marker(
    nasa_coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
        )
    )
site_map.add_child(circle)
site_map.add_child(marker)


# In[ ]:




