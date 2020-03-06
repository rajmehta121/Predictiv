#!/usr/bin/env python
# coding: utf-8

# # Taxi Demand Prediction Using LSTM Model

# ## Importing Libraries

# In[1]:


import geohash2 as ghash
import gpxpy.geo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras as ks
import seaborn as sns
import dask.dataframe as dd
import folium
from math import radians, degrees, cos, sin, sqrt, atan2
import pickle
import warnings
warnings.filterwarnings("ignore")
import datetime
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import simpy
 #Models
from sklearn.cluster import MiniBatchKMeans, KMeans # Clustering
import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from scipy import spatial
from collections import defaultdict


# In[2]:


from pymongo import MongoClient
from pprint import pprint
client = MongoClient(port=27017)
db=client.testdb
serverStatusResult=db.command("serverStatus")
pprint(serverStatusResult)


# ## Reading Dataset

# In[3]:


dt = pd.read_csv("yellow_tripdata_2015-01.csv")
dt["pickup_date"] = dt["tpep_pickup_datetime"].apply(lambda x:x.split(" ")[0]) #extracting date from datetime
#adding column for week number
dt["week"] = dt["pickup_date"].apply(lambda x:1 if x<"2015-01-08" else (2 if ((x>="2015-01-08") &                                                                               (x< "2015-01-15")) else (3 if ((x>="2015-01-15") &                                                                                                       (x< "2015-01-22")) else (4 if ((x>="2015-01-22") &                                                                                                                         (x< "2015-01-29")) else 5))))


# ## Calculating Number Of Outliers

# In[4]:


outlier_locations = dt[((dt.pickup_longitude <= -74.15) | (dt.pickup_latitude <= 40.5774)|                    (dt.pickup_longitude >= -73.7004) | (dt.pickup_latitude >= 40.9176))]
len(outlier_locations)


# In[5]:


type(dt["tpep_dropoff_datetime"][10])


# In[6]:


dt.head()


# ## Conversion from ISO time format to Unix time format

# In[7]:


def convert_to_unix(s):
    return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())


# ## Calculation of trip duration and speed of taxi during the trip

# In[8]:


def return_with_trip_times(dataset):
    duration = dt[['tpep_pickup_datetime','tpep_dropoff_datetime']]
    # pickups and dropoffs to unix time
    duration_pickup = [convert_to_unix(x) for x in duration['tpep_pickup_datetime'].values]
    duration_drop = [convert_to_unix(x) for x in duration['tpep_dropoff_datetime'].values]
    # calculate duration of trips in minutes
    durations = (np.array(duration_drop) - np.array(duration_pickup))/float(60)

    new_frame = dataset[['passenger_count','trip_distance','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','total_amount','pickup_date','week']]
    
    # append durations of trips and speed in miles/hr to a new dataframe
    new_frame['trip_times'] = durations
    new_frame['pickup_times'] = duration_pickup ## Used for time binning later
    new_frame['Speed'] = 60*(new_frame['trip_distance']/new_frame['trip_times'])
    new_frame['pickup_date_time'] = dt['tpep_pickup_datetime']
    new_frame['dropoff_date_time'] = dt['tpep_dropoff_datetime']
    return new_frame


# In[9]:


frame_with_durations = return_with_trip_times(dt)
frame_with_durations.head(10)


# ## Outlier Removal(Trip Duration)

# In[16]:


var = frame_with_durations["trip_times"].values
var = np.sort(var)
#calculating 0-100th percentile to find a the correct percentile value for removal of outliers
for i in range(0,100,10):
    print("{} percentile value is {}".format(i,var[int(len(var)*(i/100))]))
print ("100 percentile value is ",var[-1])


# In[17]:


frame_with_durations_modified = frame_with_durations[(frame_with_durations.trip_times>1) & (frame_with_durations.trip_times<720)] #trip duration <= 12 hrs


# In[18]:


frame_with_durations_modified['Speed'] = 60*(frame_with_durations_modified['trip_distance']/frame_with_durations_modified['trip_times'])


# In[19]:


var =frame_with_durations_modified["Speed"].values
var = np.sort(var)


# ## Outlier Removal(Speed)

# In[20]:



# calculating speed values at each percntile 0,10,20,30,40,50,60,70,80,90,100 
for i in range(0,100,10):
    print("{} percentile value is {}".format(i,var[int(len(var)*(i/100))]))
print("100 percentile value is ",var[-1])


# In[21]:


for i in range(90,100):
    print("{} percentile value is {}".format(i,var[int(len(var)*(i/100))]))
print("100 percentile value is ",var[-1])


# In[22]:


#calculating speed values at each percntile 99.0,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100
for i in np.arange(0.0, 1.0, 0.1):
    print("{} percentile value is {}".format(99+i,var[int(len(var)*(float(99+i)/100))]))
print("100 percentile value is ",var[-1])


# In[23]:


frame_with_durations_modified=frame_with_durations[(frame_with_durations.Speed>0) & (frame_with_durations.Speed<45.37)]


# In[24]:


#calculating average speed of taxis
sum(frame_with_durations_modified["Speed"]) / float(len(frame_with_durations_modified["Speed"]))


# ## Outlier Removal(Trip Distance)

# In[25]:


# Sorting the "trip_distance" column
var =frame_with_durations_modified["trip_distance"].values
var = np.sort(var)


# In[26]:


# calculating trip distance values at each percntile 0,10,20,30,40,50,60,70,80,90,100 
for i in range(0,100,10):
    print("{} percentile value is {}".format(i,var[int(len(var)*(i/100))]))
print("100 percentile value is ",var[-1])


# In[27]:


# calculating trip distance values at each percntile 90,91,92,93,94,95,96,97,98,99,100
for i in range(90,100):
    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))
print("100 percentile value is ",var[-1])


# In[28]:


# calculating trip distance values at each percntile 99.0,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100
for i in np.arange(0.0, 1.0, 0.1):
    print("{} percentile value is {}".format(99+i,var[int(len(var)*(float(99+i)/100))]))
print("100 percentile value is ",var[-1])


# In[29]:


frame_with_durations_modified=frame_with_durations[(frame_with_durations.trip_distance>0) & (frame_with_durations.trip_distance<22)]


# ## Outlier Removal(Total Amount)

# In[30]:


## Sort the total_amount column
var = frame_with_durations_modified["total_amount"].values
var = np.sort(var)


# In[31]:


# calculating total fare amount values at each percntile 0,10,20,30,40,50,60,70,80,90,100 
for i in range(0,100,10):
    print("{} percentile value is {}".format(i,var[int(len(var)*(i/100))]))
print("100 percentile value is ",var[-1])


# In[32]:


# calculating total fare amount values at each percntile 90,91,92,93,94,95,96,97,98,99,100
for i in range(90,100):
    print("{} percentile value is {}".format(i,var[int(len(var)*(i/100))]))
print("100 percentile value is ",var[-1])


# In[33]:


# calculating total fare amount values at each percntile 99.0,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100
for i in np.arange(0.0, 1.0, 0.1):
    print("{} percentile value is {}".format(99+i,var[int(len(var)*(float(99+i)/100))]))
print("100 percentile value is ",var[-1])


# In[34]:


frame_with_durations_modified=frame_with_durations[(frame_with_durations.total_amount>0) & (frame_with_durations.total_amount<88)]


# In[35]:


frame_with_durations_modified.shape


# In[36]:


# removing all outliers based on our univariate analysis above
def remove_outliers(new_frame):

    a = new_frame.shape[0]
    print ("Number of pickup records = ",a)
    temp_frame = new_frame[((new_frame.dropoff_longitude >= -74.15) & (new_frame.dropoff_longitude <= -73.7004) &                       (new_frame.dropoff_latitude >= 40.5774) & (new_frame.dropoff_latitude <= 40.9176)) &                        ((new_frame.pickup_longitude >= -74.15) & (new_frame.pickup_latitude >= 40.5774)&                        (new_frame.pickup_longitude <= -73.7004) & (new_frame.pickup_latitude <= 40.9176))]
    b = temp_frame.shape[0]
    print ("Number of outlier coordinates lying outside NY boundaries:",(a-b))

    
    temp_frame = new_frame[(new_frame.trip_times > 0) & (new_frame.trip_times < 720)]
    c = temp_frame.shape[0]
    print ("Number of outliers from trip times analysis:",(a-c))
    
    
    temp_frame = new_frame[(new_frame.trip_distance > 0) & (new_frame.trip_distance < 22)]
    d = temp_frame.shape[0]
    print ("Number of outliers from trip distance analysis:",(a-d))
    
    temp_frame = new_frame[(new_frame.Speed <= 45.37) & (new_frame.Speed >= 0)]
    e = temp_frame.shape[0]
    print ("Number of outliers from speed analysis:",(a-e))
    
    temp_frame = new_frame[(new_frame.total_amount <88) & (new_frame.total_amount >0)]
    f = temp_frame.shape[0]
    print ("Number of outliers from fare analysis:",(a-f))
    
    new_frame = new_frame[((new_frame.dropoff_longitude >= -74.15) & (new_frame.dropoff_longitude <= -73.7004) &                       (new_frame.dropoff_latitude >= 40.5774) & (new_frame.dropoff_latitude <= 40.9176)) &                        ((new_frame.pickup_longitude >= -74.15) & (new_frame.pickup_latitude >= 40.5774)&                        (new_frame.pickup_longitude <= -73.7004) & (new_frame.pickup_latitude <= 40.9176))]
    
    new_frame = new_frame[(new_frame.trip_times > 0) & (new_frame.trip_times < 720)]
    new_frame = new_frame[(new_frame.trip_distance > 0) & (new_frame.trip_distance < 23)]
    new_frame = new_frame[(new_frame.Speed < 45.31) & (new_frame.Speed > 0)]
    new_frame = new_frame[(new_frame.total_amount <88) & (new_frame.total_amount >0)]
    
    print ("Total outliers removed",a - new_frame.shape[0])
    print ("---")
    return new_frame
print ("Removing outliers in the month of Jan-2015")
print ("----")
frame_with_durations_outliers_removed = remove_outliers(frame_with_durations)
print("fraction of data points that remain after removing outliers", float(len(frame_with_durations_outliers_removed))/len(frame_with_durations))
    


# In[37]:


frame_order = frame_with_durations.sort_values("pickup_longitude", axis=0, ascending=True)
frame_order.head(10)


# In[38]:


X= frame_with_durations_outliers_removed[(frame_with_durations_outliers_removed["pickup_longitude"] == -77.158493) & 
                                         (frame_with_durations_outliers_removed["pickup_latitude"] == 46.704201)]
print(X)


# In[39]:


frame_with_durations_outliers_removed


# ## Clustering Using K-Means

# In[40]:


# Create a dataset containing only pickup_latitude and pickup_longitude of all the data points
# This will be used to find clusters (regions) in the New York city
coords = frame_with_durations_outliers_removed[['pickup_latitude', 'pickup_longitude']].values
coords[:10,:]


# In[41]:


# trying different cluster sizes to choose the right K in K-means

def find_min_distance(cluster_centers, cluster_len):
    less2 = [] # less2[i] => No. of clusters within the vicinity of 2 miles from cluster i 
    more2 = [] # more2[i] => No. of clusters outside the vicinity of 2 miles from cluster i
    
    min_dist=1000  ## Randomly initialize high value (like infinity)
    
    for i in range(0, cluster_len):  ## i iterates for each cluster
        nice_points = 0
        wrong_points = 0
        
        for j in range(0, cluster_len): ## j iterates for each cluster
            if j!=i:  ## For two separate clusters
                ## distance between cluster centers of clusters i and j (inter cluster distance)
                distance = gpxpy.geo.haversine_distance(cluster_centers[i][0], cluster_centers[i][1],cluster_centers[j][0], cluster_centers[j][1])
                
                ## distance is calculaed in meters and is converted to miles below
                min_dist = min(min_dist,distance/(1.60934*1000))  ## 1 mile = 1.60934 km
                
                if (distance/(1.60934*1000)) <= 2:
                    nice_points +=1
                else:
                    wrong_points += 1
        
        less2.append(nice_points)
        more2.append(wrong_points)
    
    print ("On choosing a cluster size of ",cluster_len,
           "\nAvg. Number of Clusters within the vicinity (i.e. intercluster-distance < 2):", \
           np.round(sum(less2)/len(less2), 2), "\nAvg. Number of Clusters outside the vicinity \
           (i.e. intercluster-distance > 2):", np.round(sum(more2)/len(more2), 2),\
           "\nMin inter-cluster distance = ",\
           min_dist,"\n---")

def find_clusters(increment):
    kmeans = MiniBatchKMeans(n_clusters=increment, batch_size=10000,random_state=42)
    kmeans.fit(coords)
    cluster_centers = kmeans.cluster_centers_  ## Coordinates of cluster centers
    cluster_len = len(cluster_centers)  ## No. of clusters => n_clusters
    return cluster_centers, cluster_len

# we need to choose number of clusters so that, there are more number of cluster regions 
# that are close to any cluster center
# and make sure that the minimum inter cluster distance should not be very less
for increment in range(10, 100, 10):
    cluster_centers, cluster_len = find_clusters(increment)
    find_min_distance(cluster_centers, cluster_len)


# In[42]:


#We take an optimal value of min intercluster distance as 0.5 miles.Based on this,we get no of clusters as 30.
kmeans = MiniBatchKMeans(n_clusters=30, batch_size=10000,random_state=0)
kmeans.fit(coords)
# Predict the closest cluster each sample in dataset belongs to.
frame_with_durations_outliers_removed['pickup_cluster'] = kmeans.predict(frame_with_durations_outliers_removed[['pickup_latitude', 'pickup_longitude']])
cluster_centers = kmeans.cluster_centers_
cluster_len = len(cluster_centers)

dt_dropoff = frame_with_durations_outliers_removed[frame_with_durations_outliers_removed["dropoff_date_time"] >= "2015-01-31 23:40:00"]
dt_pickup = frame_with_durations_outliers_removed[frame_with_durations_outliers_removed["pickup_date_time"] >= "2015-01-31 23:40:00"]
dt_pickup_time=np.array(dt_pickup["pickup_date_time"])
dt_dropoff_time=np.array(dt_dropoff["dropoff_date_time"])
dt_pickup_cluster = np.array(dt_pickup['pickup_cluster'])
#dt_dropoff_cluster = np.array(dt_dropoff['pickup_cluster'])
pickup_time_cluster = {}
dropoff_time_cluster = {}

#print(pickup_time_cluster)



# In[ ]:





# In[40]:


dt_dropoff_longitude = np.array(dt_dropoff['dropoff_longitude'])
dt_dropoff_latitude = np.array(dt_dropoff['dropoff_latitude'])
dt_pickup_latitude = np.array(dt_pickup['pickup_latitude'])
dt_pickup_longitude = np.array(dt_pickup['pickup_longitude'])
dt_kmean_temp = kmeans.predict(dt_dropoff[["dropoff_latitude","dropoff_longitude"]])
for i in range(0, len(dt_dropoff)):
    document = db.driver_locations.insert_one({"time":str(dt_dropoff_time[i]), "cluster_id":str(dt_kmean_temp[i]),"latitude":str(dt_dropoff_latitude[i]), "longitude":str(dt_dropoff_longitude[i])})
print(db.driver_locations.find().count())


# In[98]:


print(dt_dropoff_latitude)


# In[38]:


frame_with_durations_outliers_removed['pickup_cluster'].head(10)


# In[39]:


# Plotting the cluster centers on OSM
map_osm = folium.Map(location=[40.734695, -73.990372], tiles='Stamen Toner')
for i in range(cluster_len):
    folium.Marker(list((cluster_centers[i][0],cluster_centers[i][1])), popup=(str(cluster_centers[i][0])+str(cluster_centers[i][1]))).add_to(map_osm)
map_osm


# In[40]:


# Visualising the clusters on a map
def plot_clusters(frame):
    city_long_border = (-74.03, -73.75)  # X-axis limits
    city_lat_border = (40.63, 40.85)  # Y-axis limits
    fig, ax = plt.subplots(ncols=1, nrows=1)
    
    # Create a scatter plot of first 100000 data points with longitude on x-axis and latitude on y-axis
    # Parameter 'c' => Points belonging to the same cluster must have the same color
    # Parameter 'lw' => lw=0 means linewidth is 0.
    ax.scatter(frame.pickup_longitude.values[:100000], frame.pickup_latitude.values[:100000], s=10, lw=0,
               c=frame.pickup_cluster.values[:100000], cmap='tab20', alpha=0.2)
    ax.set_xlim(city_long_border)
    ax.set_ylim(city_lat_border)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()

plot_clusters(frame_with_durations_outliers_removed)


# ## Time BInning

# In[41]:


unix_pickup_times=[i for i in frame_with_durations_outliers_removed['pickup_times'].values]
unix_pickup_times[:5]


# In[42]:


twentyminutewise_binned_unix_pickup_times = [int((i-1420070400)/1200) for i in unix_pickup_times]
print("Min = {}, Max = {}".format(min(twentyminutewise_binned_unix_pickup_times), max(twentyminutewise_binned_unix_pickup_times)))


# In[43]:


twentyminutewise_binned_unix_pickup_times = [i+16 for i in twentyminutewise_binned_unix_pickup_times] #we add 16 as input time is according to EST(Eastern Standard Time)
print("Min = {}, Max = {}".format(min(twentyminutewise_binned_unix_pickup_times), max(twentyminutewise_binned_unix_pickup_times)))


# In[44]:


frame_with_durations_outliers_removed['pickup_bins'] = np.array(twentyminutewise_binned_unix_pickup_times)


# In[45]:


frame_with_durations_outliers_removed.columns


# In[46]:


jan_2015_frame = frame_with_durations_outliers_removed.copy()


# In[47]:


#cleaning memory by removing data frames which are no longer needed
del frame_with_durations_outliers_removed
del unix_pickup_times
del twentyminutewise_binned_unix_pickup_times


# In[48]:


jan_2015_frame.head(10)


# In[49]:


def add_pickup_bins(frame,month,year):
    unix_pickup_times=[i for i in frame['pickup_times'].values]
    ## Below are unix timestamps for Jan 01 2015, Feb 01 2015, and so on..
    unix_times = [[1420070400,1422748800,1425168000,1427846400,1430438400,1433116800],                    [1451606400,1454284800,1456790400,1459468800,1462060800,1464739200]]
    
    start_pickup_unix=unix_times[year-2015][month-1]
    tenminutewise_binned_unix_pickup_times=[(int((i-start_pickup_unix)/1200)+16) for i in unix_pickup_times]
    frame['pickup_bins'] = np.array(tenminutewise_binned_unix_pickup_times)
    return frame


# In[50]:


jan_2015_frame.head(10)


# ## Calculation of No of Trips 

# In[51]:


c = jan_2015_frame.groupby(['pickup_cluster','pickup_bins']).trip_distance.agg({'no_of_trips': 'count'})
#for key, item in jan_2015_groupby:
#    print(jan_2015_groupby.get_group(key), "\n\n")
print(c)


# In[52]:


# Gets the unique bins where pickup values are present for each reigion

# for each cluster region we will collect all the indices of 20-min intervals in which the pickups happened 
# we got an observation that there are some pickup_bins that do not have any pickups
def return_unq_pickup_bins(frame):
    values = []
    for i in range(0,30):
        new = frame[frame['pickup_cluster'] == i]
        list_unq = list(set(new['pickup_bins']))
        list_unq.sort()
        values.append(list_unq)
    return values

jan_2015_unique = return_unq_pickup_bins(jan_2015_frame)

total = 0
for i in range(30):
    total += (2232 - len(jan_2015_unique[i]))
print("Total no. of bins (across all clusters) with zero pickups in Jan 2015 = ",total)


# In[53]:


# for each cluster number of 20-min intervals with 0 pickups
print("Below is the list of no. of zero pickups in each cluster of Jan 2015")
print('*'*70)
print('*'*70)
for i in range(30):
    print("For",i,"th cluster, number of 20 min intervals with zero pickups: ",2232 - len(jan_2015_unique[i]))
    print('-'*60)


# In[54]:


# Fills a value of zero for every bin where no pickup data is present 
# the count_values: number of pickps that are happened in each region for each 20min interval
# there wont be any value if there are no pickups.
# values: number of unique bins

# for every 20min interval(pickup_bin) we will check it is there in our unique bin,
# if it is there we will add the count_values[index] to smoothed data
# if not we add 0 to the smoothed data
# we finally return smoothed data
jan_2015_groupby=jan_2015_frame.groupby(['pickup_cluster','pickup_bins']).trip_distance.agg({'no_of_trips': 'count'})
def fill_missing(count_values,values):
    smoothed_regions=[]
    ind=0  # ind iterates over count_values only
    for r in range(0,30):
        smoothed_bins=[]
        for i in range(2232):
            if i in values[r]:
                smoothed_bins.append(count_values[ind])
                ind+=1
            else:
                smoothed_bins.append(0)
        smoothed_regions.extend(smoothed_bins)
    return smoothed_regions
jan_2015_smooth = fill_missing(jan_2015_groupby['no_of_trips'].values,jan_2015_unique)
print("number of 20 min intervals among all the clusters ",len(jan_2015_smooth))


# ## Adding Geohash Value for each pair of coordinates

# In[55]:


gh = []
for i in range(0,jan_2015_frame.shape[0]):
    longi = jan_2015_frame["pickup_longitude"].iloc[i] 
    lati = jan_2015_frame["pickup_latitude"].iloc[i]
    gh.append(ghash.encode(lati,longi,precision=5))


# In[56]:


jan_2015_frame["geohash"] = gh
jan_2015_frame.head(10)


# ## Data Preparation For Model Training and Testing and Model Training

# In[57]:


#data for cluster with index 0 is grouped based on cluster and bin index and no of pickups is estimated per bin
X=jan_2015_frame[jan_2015_frame["pickup_cluster"]==0]
X_1=X.groupby(['pickup_cluster','pickup_bins']).trip_distance.agg({'no_of_trips': 'count'})
print(X_1)


# In[ ]:


#80% data for training and 20% data for testing
predictions={}
for a in range(0,30):
    print(a)
    X=jan_2015_smooth[(2232*a):(2231*(a+1))]
    labels = []    
    features_set=[]
    test_features=[]
    test_labels=[]
    for i in range((2232*a)+5,(2232*a)+int(0.8*2231)):
        features_set.append(jan_2015_smooth[i-5:i]) 
        labels.append(jan_2015_smooth[i]) 
    for i in range((2232*a)+int(0.8*2231),(2232*a)+2231):
        test_features.append(jan_2015_smooth[i-5:i]) 
        test_labels.append(jan_2015_smooth[i]) 

    features_set, labels = np.array(features_set), np.array(labels)
    test_features=np.array(test_features)
    test_features=np.reshape(test_features,(test_features.shape[0],test_features.shape[1],1))#3 dimensional data needed as input for LSTM Model
    test_labels=np.array(test_labels)
    test_labels=np.reshape(test_labels,(test_labels.shape[0],1))
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))#3 dimensional data needed as input for LSTM Model
    labels= np.reshape(labels,(labels.shape[0],1))
    model = Sequential()
    model.add(LSTM(units=250, return_sequences=True, input_shape=(features_set.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=250))
    model.add(Dropout(0.2))

    model.add(Dense(units = 1))#only single output value needed so units=1
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    history=model.fit(features_set,labels, epochs = 100, batch_size = 32)
    predictions[a] = model.predict(test_features)


# In[ ]:


print(predictions[29])


# In[81]:


b={}
for i in range(0,30):
    a=predictions[i]
    b[i]=a[len(a)-1]
print(b[1][0])
print(b)


# In[13]:


#simualation code



def simulationFor20Minutes(env):
    count = 1
    while True:
        #code for mapping
        time_lower_limit = convert_to_unix("2015-01-31 23:40:00") + count-1 * 300
        time_upper_limit = convert_to_unix("2015-01-31 23:40:00") + count * 300
        cursor_of_requests = db.request_locations.find()
        i = 0
        for x in cursor_of_requests:
            print(x)
            if cursor_of_requests[x]["time"] >= time_lower_limit and cursor_of_requests[x]["time"] < time_upper_limit:
                DC[i] = (int)cursor_of_requests[x]["cluster_id"]
                i += 1
        count += 1
        yield(env.timeout(300000))


# In[ ]:


env = simpy.Environment()
env.process(simulationFor20Minutes(env))
env.run(until = 1200000000000000000)


# In[68]:


for a in range(0,30):
    print(a)
    X=jan_2015_smooth[(2232*a):(2231*(a+1))]
    labels = []    
    features_set=[]
    test_features=[]
    test_labels=[]
    for i in range((2232*a)+5,(2232*a)+int(0.8*2231)):
        features_set.append(jan_2015_smooth[i-5:i]) 
        labels.append(jan_2015_smooth[i]) 
    for i in range((2232*a)+int(0.8*2231),(2232*a)+2231):
        test_features.append(jan_2015_smooth[i-5:i]) 
        test_labels.append(jan_2015_smooth[i]) 

    features_set, labels = np.array(features_set), np.array(labels)
    test_features=np.array(test_features)
    test_features=np.reshape(test_features,(test_features.shape[0],test_features.shape[1],1))#3 dimensional data needed as input for LSTM Model
    test_labels=np.array(test_labels)
    test_labels=np.reshape(test_labels,(test_labels.shape[0],1))
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))#3 dimensional data needed as input for LSTM Model
    labels= np.reshape(labels,(labels.shape[0],1))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    scores = model.evaluate(features_set,labels, verbose=0)
    print(scores)
    
    #for score in scores:
    #    print(score)
    print("%s: %.2f%%" % (model.metrics_names[1] ,scores[1]*100))


# In[69]:


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# In[70]:


from keras.models import model_from_json
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
loaded_model = model_from_json(open('model.json').read(),{'Dense': Dense})
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# In[73]:


X=jan_2015_frame[jan_2015_frame["pickup_cluster"]==29]
X_1=X.groupby(['pickup_cluster','pickup_bins']).trip_distance.agg({'no_of_trips': 'count'})
X_1.shape
labels = []    
features_set=[]
test_features=[]
test_labels=[]
for i in range(5,int(0.8*len(X_1))):
    features_set.append(X_1["no_of_trips"].iloc[i-5:i]) 
    labels.append(X_1["no_of_trips"].iloc[i]) 
for i in range(int(0.8*len(X_1)),len(X_1)):
    test_features.append(X_1["no_of_trips"].iloc[i-5:i]) 
    test_labels.append(X_1["no_of_trips"].iloc[i]) 

features_set, labels = np.array(features_set), np.array(labels)
test_features=np.array(test_features)
test_features=np.reshape(test_features,(test_features.shape[0],test_features.shape[1],1))#3 dimensional data needed as input for LSTM Model
test_labels=np.array(test_labels)
test_labels=np.reshape(test_labels,(test_labels.shape[0],1))
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))#3 dimensional data needed as input for LSTM Model
labels= np.reshape(labels,(labels.shape[0],1))
print(loaded_model.predict(test_features))


# ## Model Training

# ## Model Testing(Prediction)

# ## Performance Visualization

# In[75]:


for a in range(0,30):
    print(a)
    X=jan_2015_smooth[(2232*a):(2231*(a+1))]
    labels = []    
    features_set=[]
    test_features=[]
    test_labels=[]
    for i in range((2232*a)+5,(2232*a)+int(0.8*2231)):
        features_set.append(jan_2015_smooth[i-5:i]) 
        labels.append(jan_2015_smooth[i]) 
    for i in range((2232*a)+int(0.8*2231),(2232*a)+2231):
        test_features.append(jan_2015_smooth[i-5:i]) 
        test_labels.append(jan_2015_smooth[i]) 

    features_set, labels = np.array(features_set), np.array(labels)
    test_features=np.array(test_features)
    test_features=np.reshape(test_features,(test_features.shape[0],test_features.shape[1],1))#3 dimensional data needed as input for LSTM Model
    test_labels=np.array(test_labels)
    test_labels=np.reshape(test_labels,(test_labels.shape[0],1))
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))#3 dimensional data needed as input for LSTM Model
    labels= np.reshape(labels,(labels.shape[0],1))
    plt.figure(figsize=(10,6))
    plt.plot(test_labels, color='blue', label='Actual No of Pickups')
    plt.title('Taxi Demand Data')
    plt.xlabel('Time Bin')
    plt.ylabel('No Of Pickups')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(predictions[a] , color='red', label='Predicted No of Pickups')
    plt.title('Taxi Demand Prediction')
    plt.xlabel('Time Bin')
    plt.ylabel('No Of Pickups')
    plt.legend()
    plt.show()


# In[ ]:


mse=mean_squared_error(test_labels, predictions)
print(mse/1000)


# In[ ]:


#scores = model.evaluate(test_features,test_labels,verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


#scores = model.evaluate(test_features,test_labels,verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


#longitude=0.4496
#latitude=0.3402


# In[ ]:


import gmaps
import gmaps.datasets
import os


# In[ ]:


#gmaps.configure(api_key='AI...')
#export GOOGLE_API_KEY=AIzaSyCOdkx0uMZUPnK7A2ONxT7w5ZjobwrwIEU
#gmaps.configure(api_key="AIzaSyCOdkx0uMZUPnK7A2ONxT7w5ZjobwrwIEU")
#gmaps.figure()


# In[ ]:


new_york_coordinates = (40.75, -74.00)
gmaps.figure(center=new_york_coordinates, zoom_level=12)


# In[ ]:


d_database = pd.read_csv("driver database.csv")
d_database.head(10)
arr=kmeans.predict(d_database[["pickup_latitude","pickup_longitude"]])
arr=arr[:100]
print(arr)


# In[76]:

    def makeKeyR(p,q):
        if len(str(p)) == 1:
            return "0"+str(p)+str(q)
        if len(str(q)) == 1:
            return str(p)+"0"+str(q)
        if len(str(p)) != 1 and len(str(q)) != 1:
            return str(p)+str(q)        
    def makeKeyD(a,b):
        if len(str(a)) == 1:
            return str(b)+"0"+str(a)
        if len(str(b)) == 1:
            return "0"+str(b)+str(a)
        if len(str(a)) != 1 and len(str(b)) != 1:
            return str(b)+str(a)

cursor_of_drivers = db.driver_locations.find()
countc = 1
DC=[]
time_lower_limit={}
time_upper_limit={}
time_lower_limit[0] = convert_to_unix("2015-01-31 23:40:00") #+ countc-1 * 300
time_upper_limit[0] = convert_to_unix("2015-01-31 23:45:00") #+ countc * 300        
time_lower_limit[1] = convert_to_unix("2015-01-31 23:45:00") #+ countc-1 * 300
time_upper_limit[1] = convert_to_unix("2015-01-31 23:50:00") #+ countc * 300        
time_lower_limit[2] = convert_to_unix("2015-01-31 23:50:00") #+ countc-1 * 300
time_upper_limit[2] = convert_to_unix("2015-01-31 23:55:00") #+ countc * 300        
time_lower_limit[3] = convert_to_unix("2015-01-31 23:55:00") #+ countc-1 * 300
time_upper_limit[3] = convert_to_unix("2015-01-31 23:59:59") #+ countc * 300        

driver_id = 0
driver_longitude_dict = {}
driver_latitude_dict = {}
while True:
    cursor_of_drivers = db.driver_locations.find()
    for x in cursor_of_drivers:
        if convert_to_unix(x["time"]) >= time_lower_limit[countc-1] and convert_to_unix(x["time"]) < time_upper_limit[countc-1]:
            DC.append(int(x["cluster_id"],10))
            driver_longitude_dict[driver_id] = float(x["longitude"])
            driver_latitude_dict[driver_id] = float(x["latitude"])
            driver_id += 1
    requests=[0,7,2,3,5,2,0,4,5,4,2,3,0,7,5,3,2,6,3,9,3,1,3,4,7,3,2,4,0,1]
    boxplot = {}


    #print(DC)




    unique, counts = np.unique(DC, return_counts=True)
    dri = dict(zip(unique, counts))
    drivers=np.zeros(30)
    for i in range(0,30):
        if i in dri:
            drivers[i] = dri[i]
    #print(dri)

    mapped_drivers={}
    rank_table={}
    rank_R={}
    rank_D={}
    xyz={}

    for i in range(0,30):
        if requests[i] >= drivers[i] :
            requests[i] = requests[i] - drivers[i]
            drivers[i] = 0
        else:
            drivers[i] = drivers[i] - requests[i]
            requests[i] = 0
    #print(requests)
    #print(drivers)
    abc = {}
    for i in range(0,30):
        if requests[i] >=1:
            inter_cluster_rank = {}
            latitude = cluster_centers[i][0]
            longitude = cluster_centers[i][1]
            for j in range(0,len(DC)):
                if drivers[DC[j]]>0:
                    driver_longitude = driver_longitude_dict[j]
                    driver_latitude = driver_latitude_dict[j]
                    dLon = radians(driver_longitude - longitude)
                    dLat = radians(driver_latitude - latitude)
                    lat1 = radians(latitude)
                    lat2 = radians(driver_latitude)
                    a = sin(dLat/2)*sin(dLat/2) + sin(dLon/2)*sin(dLon/2)*cos(lat1)*cos(lat2)
                    c = 2*atan2(sqrt(a),sqrt(1-a))
                    haversine = c*6371000
                    if len(str(i)) == 1:
                        rank_R["0"+str(i)+str(j)] = haversine
                        rank_D[str(j)+"0"+str(i)] = haversine
                    if len(str(j)) == 1:
                        rank_R[str(i)+"0"+str(j)] = haversine
                        rank_D["0"+str(j)+str(i)] = haversine
                    if len(str(i)) != 1 and len(str(j)) != 1:
                        rank_R[str(i)+str(j)] = haversine
                        rank_D[str(j)+str(i)] = haversine
                    inter_cluster_rank[makeKeyR(i,j)] = rank_R[makeKeyR(i,j)]
            abc.update{k: v for k, v in sorted(inter_cluster_rank.items(), key=lambda item: item[1])}
      
    pqr = {k: v for k, v in sorted(rank_D.items(), key=lambda item: item[1])}
    #print(pqr)
    #print(pqr.keys)
    list_values_R = [ v for v in abc.keys() ]
    list_rank_R = [i for i in range(1,len(abc)+1)]
    ranked_dict_R = dict(zip(list_values_R, list_rank_R))
    list_values_D = [ v for v in pqr.keys() ]
    list_rank_D = [i for i in range(1,len(pqr)+1)]
    ranked_dict_D = dict(zip(list_values_D,list_rank_D))
    #print(ranked_dict_D)
    count=0
    for i in range(0,30):
        for j in range(0,len(DC)):
            if (str(i)+str(j)) in ranked_dict_R and (str(j)+str(i)) in ranked_dict_D:
                if len(str(i)) == 1:
                    R= ranked_dict_R["0"+str(i)+str(j)] + ranked_dict_D[str(j)+"0"+str(i)]
                    rank_table[str(i)+str(j)] = R
                if len(str(j)) == 1:
                    R= ranked_dict_R[str(i)+"0"+str(j)] + ranked_dict_D["0"+str(j)+str(i)]
                    rank_table[str(i)+str(j)] = R
                if len(str(i)) != 1 and len(str(j)) != 1:
                    R= ranked_dict_R[str(i)+str(j)] + ranked_dict_D[str(j)+str(i)]
                    rank_table[str(i)+str(j)] = R
    
    
    sorted_rank_table = {k: v for k, v in sorted(rank_table.items(), key=lambda item: item[1])}
    list_rank = [ v for v in sorted_rank_table.keys() ]
    l = 0
    
    for k in range(0,len(sorted_rank_table)):
        a = list_rank[k] 
        i = a[:2]
        j = a[2:]
       # i = int(i,10)
        #j=float(j)
       # print(i)
        #print(j)
        
        if requests[int(i,10)]>0 and drivers[DC[int(j,10)]]>0:
            requests[int(i,10)]=requests[int(i,10)]-1
            DC[int(j,10)]=int(i,10)
            boxplot[a] = rank_R[a]
            
            
            
    list_values_boxplot = [ v for v in boxplot.values() ]
    #arr_val_box=np.array(list_values_boxplot)
    arr_values_boxplot=[]
    for i in range(0,len(list_values_boxplot)):
        arr_values_boxplot.append(list_values_boxplot[i])
    print(countc)
    t = np.percentile(arr_values_boxplot, 75)    
    print(arr_values_boxplot)
    d=""
    for item in boxplot:
        j = int(item[2:],10)
        if boxplot[item] <= t:
            DC[int(item[2:],10)] = int(item[:2],10)
        else:
            centroid = cluster_centers[int(item[:2],10)]
            lat = driver_latitude_dict[int(item[:2],10)]
            lon = driver_longitude_dict[int(item[:2],10)]
            while True:
                lat = lat + t
                lon = lon + t
                if lat > centroid[0]:
                    lat = centroid[0]
                if lon > centroid[1]:
                    lon = centroid[1]
                c_id = int(kmeans.predict([[lat,lon]]))
                DC[int(item[2:],10)] = c_id
                for d_id in range(0,len(DC)):
                    if len(str(d_id)) == 1:
                        d = "0" + str(d_id)
                    if DC[d_id] == int(item[:2],10) and (item[:2]+d) not in boxplot.keys():
                        y = d_id
                        break
                j=y
                m=DC[j]
                if m == int(item[:2],10):
                    break
    print(DC)
    for i in range(0,len(DC)):
        document = {"no":str(i),"driver_lat":str(driver_latitude_dict[i]),"driver_long":str(driver_longitude_dict[i]),"cluster_lat":str(cluster_centers[DC[i]][0]),"cluster_long":str(cluster_centers[DC[i]][1])}
        db.dc_array.insert_one(document)
    DC=[]
    countc += 1
    if countc == 5:
        break
                        
    


# In[ ]:




