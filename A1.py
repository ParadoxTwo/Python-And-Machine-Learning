#!/usr/bin/env python
# coding: utf-8

# #**SIT 720 - Machine Learning**
# 
# Lecturer: Chandan Karmakar | karmakar@deakin.edu.au
# 
# School of Information Technology,
# <br/>Deakin University, VIC 3125, Australia.

# #**Assessment Task 1 (20 marks)**
# 
# ##**Submission Instruction**
# 1.  Student should insert Python code or text responses into the cell followed by the question.
# 
# 2.  For answers regarding discussion or explanation, **maximum five sentences are suggested**.
# 
# 3.  Rename this notebook file appending your student ID. For example, for student ID 1234, the submitted file name should be A0_1234.ipynb.
# 
# 4.  Insert your student ID and name in the following cell.

# In[ ]:


# Student ID: 218599279

# Student name: Edwin John Nadarajan


# 
# 
# ##**Background**
# Selection of housing is always difficult for someone seeking for a suitable one as it includes various factors and preferences. People prefer to buy a house considering many criteria like- population, quality of life, financial capability, as well as social and natural environments around the housing block. In this assignment you will be helping people choosing suitable housing for them according to their needs.
# ##**Dataset**
# **Dataset file name:** housing_dataset.csv
# 
# **Dataset description:** Dataset contains total 10 features (columns). It contains the location, housing age, population, number of families in a housing (block), number of rooms, average income of the families in that housing, ocean proximity and other informaiton. Each row indicates a record of a housing block containing the features mentioned earlier.
# 
# **Features:** 
# 
# 1.   latitude (float): Latitude of the location of a housing in conventional geospace
# 2.   longitude (float): Longitude of the location of a housing in conventional geospace
# 3.   housing_age (int): Age of the housing in year, the higher number indicates the older housing
# 4.    total_rooms (int): Total number of rooms in a housing
# 5.    total_bedrooms (int): Total number of bedrooms in a housing
# 6.    population (int): Total population of a housing
# 7.    families (int): Total number of families living in a housing
# 8.    average_income (float): Average income of the member of a housing in a scale of Tousand Dollar Per Month
# 9.    ocean_proximity (string): Describing how close the housing is to the ocean
# 10.   house_value (int): Average individual house price of a housing in Dollers
# 

# 
# 
# ##**Part-1: Basic Calculations:**  *(8 marks: 8 questions x 1 marks each)*
# 
# 
# 1.   Find the distances of the farthest and nearest housing blocks from the house block described in the first row of the dataset.

# In[2]:


# INSERT your code here.
import pandas as pd
import math

def distance(lat1, lon1, lat2, lon2): #to calculate the distance between 2 coordinates based on radians
    R = 6371.0 #radius of earth
    dLat = lat2 - lat1
    dLon = lon2 - lon1
    a = math.sin(dLat/2)*math.sin(dLat/2)+math.cos(lat1)*math.cos(lat2)*math.sin(dLon/2)*math.sin(dLon/2)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R*c
    return d

df = pd.read_csv ('housing_dataset.csv')
lat1 = df.loc[0]['latitude']
lon1 = df.loc[0]['longitude']
nearest = 1 #storing index of the nearest house
nDistance = 2**31-1 #storing distance of nearest house
farthest = 0 #storing index of farthest house
fDistance = 0 #storing distance of farthest house
dfLength = df.size/10.0
print(dfLength)

for row in range(1,int(dfLength)):
    lat = df.loc[row]['latitude']
    lon = df.loc[row]['longitude']
    d = distance(lat1, lon1, lat, lon)
    if(d<nDistance):
        nearest = row
        nDistance = d
    if(d>fDistance):
        farthest = row
        fDistance = d
    
print("The housing block that's nearest to the 1st block is: \n"+str(df.loc[nearest])+"\nHaving a distance of "+str(nDistance))
print("The housing block that's farthest to the 1st block is: \n"+str(df.loc[farthest])+"\nHaving a distance of "+str(fDistance))
    


# 2.   Calculate the average age of the house blocks near the ocean.

# In[52]:


# INSERT your code here.
oceanDf = df.loc[df['ocean_proximity']=='NEAR OCEAN']
sumHousingAge = 0
oceanLength = oceanDf.size/10
for row in oceanDf.itertuples():
    sumHousingAge+=row.housing_age
averageHousingAge = sumHousingAge/oceanLength
print("The average age of house blocks near the ocean is: "+str(averageHousingAge))


# 3.   Find the income of the housing block with the most and least population density (per family).

# In[53]:


# INSERT your code here.
leastIndex = 0
mostIndex = 0
leastDensity = 2**31-1
mostDensity = 0
for row in df.itertuples():
    density = row.population/row.families
    if (density<leastDensity):
        leastIndex = row.Index
        leastDensity = density
    if(density>mostDensity):
        mostIndex = row.Index
        mostDensity = density
print(leastDensity,end='\n')
print("Income of the housing block with the least population density: "+str(df.loc[leastIndex]['average_income']),end='\n')
print(mostDensity,end='\n')
print("Income of the housing block with the most population density: "+str(df.loc[mostIndex]['average_income']),end='\n')


# 4.   Calculate the price difference between the latest and oldest housing block from the dataset.

# In[54]:


# INSERT your code here.
latestIndex = 0
latest = 2**31-1
oldestIndex = 0
oldest = 0
for row in df.itertuples():
    if (row.housing_age<latest):
        latestIndex = row.Index
        latest = row.housing_age
    if(row.housing_age>oldest):
        oldestIndex = row.Index
        oldest = row.housing_age
print(latest,end='\n')
print(oldest,end='\n')
print("The price difference between the latest and oldest housing block: "+str(df.loc[latestIndex]['house_value']-df.loc[oldestIndex]['house_value']),end='\n')


# 5.   Calculate the cheapest price per room from the dataset.

# In[57]:


# INSERT your code here.
cheapest = 2**31-1
for row in df.itertuples():
    price = row.house_value/row.total_rooms
    if(price<cheapest):
        cheapest = price
        
print("The cheapest price per room is: "+str(cheapest))


# 6.   Calculate the population density (per family) for the most and least wealthy housing blocks in the dataset.

# In[59]:


# INSERT your code here.
leastIndex = 0
least = 2**31-1
mostIndex = 0
most = 0
for row in df.itertuples():
    if (row.average_income<least):
        leastIndex = row.Index
        least = row.average_income
    if(row.average_income>most):
        mostIndex = row.Index
        most = row.average_income
print("Popolation Density for the least wealthy housing block: "+str(df.loc[leastIndex]['population']/df.loc[leastIndex]['families']), end='\n')
print("Popolation Density for the most wealthy housing block: "+str(df.loc[mostIndex]['population']/df.loc[mostIndex]['families']), end='\n')


# 7.   Calculate and print the total housing blocks located in the same place.

# In[77]:


# INSERT your code here.
currentLat = df.loc[0]['latitude']
currentLon = df.loc[0]['longitude']
location = str(currentLat)+", "+str(currentLon)
locationDictionary = {location:0}
for row in df.itertuples():
    if(currentLat==row.latitude and currentLon==row.longitude):
        location = str(currentLat)+", "+str(currentLon)
        val = locationDictionary[location]
        val+=1
        locationDictionary[location] = val
    else:
        currentLat=row.latitude
        currentLon=row.longitude
        location = str(currentLat)+", "+str(currentLon)
        locationDictionary[location]=1
for location in locationDictionary:
    print("Housing blocks at "+location+" are "+str(locationDictionary[location]))
    


# 8.   Calculate the price of expensive room grouped by ocean proximity.

# In[81]:


# INSERT your code here.
proximities = df['ocean_proximity'].unique()
for proximity in proximities:
    proximityDf = df.loc[df['ocean_proximity']==proximity]
    price = 0
    for row in proximityDf.itertuples():
        roomPrice = row.house_value/row.total_rooms
        if(price<roomPrice):
            price=roomPrice
    print("Most expensive room in "+proximity+" proximity is priced at: "+str(price),end='\n')


# ##**Part-2: Visualization:**  *(6 marks: 3 question x 2 marks each)*
# 
# 
# 1.   Draw the population scatter plot against housing age and another against ocean proximity. From the graph conclude an assumption.

# In[7]:


# INSERT your code here.
import matplotlib.pyplot as plot

df.plot.scatter(x='population', y='housing_age', title= "Scatter plot of population against housing age")
plot.show(block=True);
print("It is observed that older a housing block gets, fewer the population becomes therein.")
df.plot.scatter(x='population', y='ocean_proximity', title= "Scatter plot of population against ocean proximity")
plot.show(block=True);
print("It is observed that very few people prefer to stay in island housing blocks.")


# 2.  Draw a bar diagram of average values of all suitable columns. (excluding latitude, longitude and ocean proximity of course).

# In[ ]:


# INSERT your code here.
average = {
    'housing_age' : 0,
    'total_rooms' : 0,
    'total_bedrooms' : 0,
    'population' : 0,
    'families' : 0,
    'average_income' : 0,
    'house_value' : 0
}
length = df.size/10
beds = 0
for row in df.itertuples():
    average['housing_age']+=row.housing_age/length
    average['total_rooms']+=row.total_rooms/length
    if(not math.isnan(row.total_bedrooms)):
        average['total_bedrooms']+=row.total_bedrooms/length
    average['population']+=row.population/length
    average['families']+=row.families/length
    average['average_income']+=row.average_income/length
    average['house_value']+=row.house_value/length
average['average_income']*=100
average['house_value']/=100
plot.bar(range(len(average)), list(average.values()))
plot.xticks(range(len(average)), list(average.keys()))
plot.show(block=True);
print("Scaling up averaging income by 100 times and scaling down house value by 100 times for better visuals.")


# 3.   Visualize the differences in housing prices from the average price of housing using a bar diagram.

# In[19]:


# INSERT your code here.
length = df.size/10
housePriceAverage = sum(df.loc[0:length]['house_value'])/length
print(housePriceAverage)
housePricePlot = []
for row in df.itertuples():
    val = row.house_value-housePriceAverage
    housePricePlot.append(val)
data = {}
row = [x / 1.0 for x in range(int(length))]
print(len(row))
print(len(housePricePlot))
data['Housing Blocks'] = row
data['Housing Prices'] = housePricePlot
dataFrame  = pd.DataFrame(data = data);
dataFrame.plot.barh(x='Housing Prices', y='Housing Blocks', title="Differences in housing prices from the average price of housing");
plot.show(block=True);


# ##**Part-3: File Management:**  *(6 marks: 2 question x 3 marks each)*
# 
#  
# 
# 1.   Save the details of all housing blocks in a csv file having houses near oceans and  lower than the average of the housing value.

# In[16]:


# INSERT your code here.
length = df.size/10
housePriceAverage = sum(df.loc[0:length]['house_value'])/length
print(housePriceAverage)
nearOceanDf = df.loc[df['ocean_proximity']=='NEAR OCEAN']
solutionDf = nearOceanDf.loc[nearOceanDf['house_value']<housePriceAverage]
print(solutionDf)
solutionDf.to_csv(r'solution3.1.csv', index=False)


# 
# 
# 
# 
# 
# 2.   Create a new housing dataset (a csv file) having only the location, total rooms and housing price information.
# 
# 
# 
# 
# 

# In[18]:


# INSERT your code here.
df1 = df[['latitude', 'longitude', 'total_rooms', 'house_value']]
print(df1);
df1.to_csv(r'solution3.2.csv', index=False)


# In[ ]:




