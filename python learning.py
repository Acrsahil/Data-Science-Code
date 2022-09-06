#!/usr/bin/env python
# coding: utf-8

# In[90]:


## array in numpy
import numpy as np
my_lst = [1,2,3,4,5,6,]
arr = np.array(my_lst)
type(arr)


# In[97]:


##broad casting function

arr=np.array([1,2,3,4,5,6,7,8,9,10])
arr[3:]=100
arr


# In[ ]:





# In[57]:


## copy function and broadcasting function
print(arr)
array1=arr.copy()
array1[3:] = 500
array1


# In[59]:


val=2
ex1=array1[array1<100]
ex2=array1<2
ex3=arr*2
ex4=arr/2
print(ex1,ex2,"multiplication=",ex3,"division =",ex4)


# In[62]:


np.onesh(4,dtype=int)


# In[25]:


np.ones((3,3),dtype=int)


# In[93]:


import pandas as pd
import numpy as np


# In[7]:


df=pd.DataFrame(np.arange(0,20).reshape(5,4),index=['Row1','Row2','Row3','Row4','Row5'],columns=["column1","column2","column3","column4"])


# In[8]:


df.head()


# In[68]:


df.iloc[0:4,0:2]


# In[10]:


type(df.loc['Row1'])


# In[36]:


df


# In[18]:


##convert dataFrame into array
df.iloc[0:4,0:4].values


# In[23]:


#Null in inbuilt function
df.isnull().sum()


# In[38]:


df


# In[ ]:


##finding unique values 


# In[41]:


df['column1'].unique()


# In[55]:


df=pd.DataFrame(np.arange(0,20).reshape(5,4),index=['Row1','Row2','Row3','Row4','Row5'],columns=['Column1','Column2','Column3','Column4'])


# In[45]:


df.head()


# In[57]:


df=pd.DataFrame(np.arange(0,20).reshape(5,4),index=["Row1","Row2","Row3","Row4","Row5"],columns=['column1','column2','column3','column4'])


# In[58]:


df.head()


# In[59]:


df=pd.DataFrame(np.arange(0,20).reshape(5,4),index=["row1","row2","row3","row4","row5"],columns=['column1','column2','column3','column4'])


# In[60]:


df.head()


# In[7]:


import pandas as pd
import numpy as np


# In[9]:


##Reading different file 
df=pd.read_csv('car.csv')


# In[61]:


df.head()


# In[63]:


df.info()


# In[53]:


df.describe()


# In[69]:


df


# In[76]:


##Get the unique catogary counts
df['body_styles'].value_counts()


# In[75]:


##Using sep Function
test_df=pd.read_csv('test.csv',sep=',')


# In[68]:


test_df.head()


# In[77]:


df


# In[81]:


##this code is pending file not found ::df[df['body_styles']>100]b


# In[6]:


##CSV
from io import StringIO, BytesIO


# In[5]:


data=('col1,col2,col3\n'
     'x,y,1\n'
     'a,b,2\n'
     'c,d,3')


# In[13]:


print(data)


# In[132]:


StringIO()


# In[137]:


pd.read_csv(StringIO(data))


# In[31]:


##Read form specefic colums
df=pd.read_csv(StringIO(data), usecols=['col1','col3'])


# In[11]:


df


# In[17]:


##Again converting this data back into CSV
df.to_csv('Test.csv')


# In[ ]:


##Specifying colums data types 


# In[5]:


from io import StringIO, BytesIO


# In[26]:


data1=('a,b,c,d\n'
     '5,6,7,8\n'
     '9,10,11')


# In[27]:


print(data1)


# In[38]:


df1=pd.read_csv(StringIO(data1),dtype=object)


# In[39]:


df1


# In[42]:


df=pd.read_csv(StringIO(data1),dtype=object)


# In[43]:


df


# In[44]:


##Note:it will basically change the data in to string


# In[45]:


df['a']


# In[46]:


##I can also replace (a,b,c,d) in different data type.


# In[80]:


df=pd.read_csv(StringIO(data1),dtype={'b':int,'c':float,'a':int})
df


# In[78]:


type(df['a'][1])


# In[81]:


##check the datatype


# In[84]:


df.dtypes


# In[85]:


##index colums and trainining delimiters


# In[87]:


data=('index,a,b,c\n'
     '4,apple,bat,5.7\n'
     '8,orange,cow,10')


# In[89]:


pd.read_csv(StringIO(data),index_col=0)


# In[92]:


data=('a,b,c\n'
     '4,apple,bat\n'
     '8,orange,cow')


# In[93]:


pd.read_csv(StringIO(data),index_col=False)


# In[94]:


##Combining usecols and index_col


# In[95]:


data=('a,b,c\n'
     '4,apple,bat\n'
     '8,Orange,cow')


# In[96]:


pd.read_csv(StringIO(data),usecols=['b','c'],index_col=False)


# In[97]:


##Quating and Escape Character very useful in NCP


# In[101]:


data='a,b\n"hello,\\"Bob\\",nice to see you",5'


# In[102]:


pd.read_csv(StringIO(data),escapechar='\\')


# In[103]:


##Url to CSv


# In[110]:


df=pd.read_csv('https://download.bls.gov/pub/time.series/cu/cu.item',sep='\t')


# In[131]:


df.head()


# In[ ]:


##file to csv  


# In[130]:


df=pd.read_csv('big data.csv',sep='\t')


# In[132]:


df


# In[ ]:


##taking only less then 100 value of sort_sequence


# In[138]:


less=df[df['sort_sequence']<100]


# In[139]:


less


# In[140]:


##Read Json to CSV


# In[14]:


Data='{"employee_name":"James","email":"acrsahil18@gmail.com","Job_profile":[{"title1":"Team Lead","title2":"SDeveloper"}]}'


# In[22]:


data1=pd.read_json(Data)


# In[74]:


df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)


# In[75]:


df.head()


# In[1]:


#convert Json to CSV


# In[76]:


df.to_csv('wine.csv')
##Now, file is created 


# In[35]:


##Convert Json to different json formats
data1.to_json(orient="records")


# In[36]:


df.to_json(orient="records")


# In[37]:


#Reading HTML content 


# In[41]:


dfs=pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/')


# In[45]:


dfs[0]


# In[56]:


url_mcc = 'https://en.wikipedia.org/wiki/Mobile_country_code'


# In[59]:


dfs=pd.read_html(url_mcc,match='Mobile network codes',header=0)


# In[60]:


dfs[0]


# In[ ]:


##Reading Excel File


# In[78]:


df_excel=pd.read_excel('pivot_tabe.xlsx'),'sheet_name=0'


# In[46]:


df_excel


# In[48]:


##pickling
##All pandas objects are equipped to_pickle methods which use python's cpickle module to save structures to disk using the pickle format..


# In[15]:


df.to_pickle("pivot_table.")


# In[16]:


df=pd.read_pickle("pivot_table")


# In[17]:


df.head()


# # Matplot Lib Tutorial
# 
# Matplotlib is a plotting for the python programming language and its numerical extension Numpy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like TKinter, wxpython, Qt or GTK+.
# 
# Some of the mafor pros of Matplotib are:
#  * Generally easy to get started for simple plots
#  * Support for sutom labels and texts 
#  * Great control of every element in a figure 
#  * High-quality output in many formats
#  * Very customizable in general.

# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[180]:


##Simple examples
x=np.arange(0,10)
y=np.arange(11,21)


# In[26]:


##plotting using matplot lib


# In[190]:


##plot scatter
plt.scatter(x,y,c='b')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Graph in 2D')
plt.show()
plt.savefig('test.png')


# In[109]:


## Plt plot


# In[57]:


x=np.arange(0,10)
y=np.arange(10,20)


# In[60]:



plt.plot(x,y,'r*',linestyle='dashed',linewidth=2,markersize=12 )
plt.xlabel('x axis')
plt.title('2D Graph')


# In[ ]:


##Creating subplots


# In[28]:


x=np.arange(0,10)
y=np.arange(11,21)

y=x*x

plt.subplot(2,2,1)
plt.plot(x,y,'ro--')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

plt.subplot(2,2,2)
plt.plot(x,y,'b*')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

plt.subplot(2,2,3)
plt.plot(x,y,'go-')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

plt.subplot(2,2,4)
plt.plot(x,y,'yo-')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')


# In[3]:


np.pi


# In[5]:


#Compute the x and y cordinate for points on a since curve


# In[33]:


x=np.arange(0,4*np.pi,0.1)
y=np.sin(x)
plt.title("Sine wave form")

#plot the points using matplotlib
plt.plot(x,y,'s-')
plt.show()


# In[47]:


#Subplot()
#Compute x and y coordinates for points on size and zosine curves

x=np.arange(0,5*np.pi,0.1)
y_sin=np.sin(x)
y_cos=np.cos(x)

#Set up a subplot grid that has height 2 and width 1,
#And set the first such subplot as active

plt.subplot(2,1,1)
plt.plot(x,y_sin,'g*')
plt.title('Sine')

#Set the second subplot as active, and make the second plt.
plt.subplot(2,1,2)
plt.plot(x,y_cos,'r*')
plt.title('Cosine')
         


# In[61]:


x=np.arange(0,10*np.pi,0.1)
y_sin=np.sin(x)
y_cos=np.sin(x)

plt.subplot(2,1,1)
plt.plot(x,y_sin,'g*')
plt.title('Sine Graph')

plt.subplot(2,1,2)
plt.plot(x,y_cos,'r*')
plt.title('Cosme Graph')
plt.show()


# In[62]:


##Bar plot


# In[25]:


x=[10,8,5,11]
y=[11,6,15,11]
x2=[2,3,7,15]
y2=[4,8,10,11]

plt.bar(x,y,color='g')
plt.bar(x2,y2,color='g')
plt.title('Bar Graph')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')


# In[21]:


x=[2,8,10]
y=[11,16,9]

x2=[3,9,11]
y2=[3,14,7]

plt.subplot(2,1,1)
plt.bar(x,y)

plt.subplot(2,1,2)
plt.bar(x2,y2,color='b')

plt.title('Bar Graph')
plt.xlabel('X axis')
plt.ylabel('y axis')
plt.show()


# In[26]:


##Histograms


# In[27]:


a=np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
plt.hist(a)
plt.title("Histogram")
plt.show()


# In[53]:


import numpy as np
import matplotlib.pyplot as plt


# In[69]:


##Box plot using Matplot
data=[np.random.normal(0,std,100) for std in range(1,4)]

#rectangle box plot 
plt.boxplot(data,vert=True,patch_artist=True);


# In[18]:


data


# In[4]:


#Data to plot


# In[48]:


labels='pyton','C++','Ruby','Java'
sizes=[215,130,245,210]
colors=['gold','yellow','green','lightcoral','lightskyblue']
explode=(0.3,0,0,0) #explode 1st slice

#plot
plt.pie(sizes,explode,labels,colors,autopct='%1.1f%%',shadow=True)

plt.show()


# In[52]:


labels='nepal','china','bhutan','india'
sizes=[250,180,290,345]
colors=['skyblue','green','yellow','red']
explode=(0.3,0,0,0)
plt.pie(sizes,explode,labels,colors,autopct='%1.1f%%',shadow=True)
plt.show()


# # Seaborn Tutorial
# ## distribution plots
# 
# * distplot
# * join plot
# * pairplot
# 
# practise problem on IRIS Dataset

# In[1]:


import seaborn as sns


# In[4]:


df=sns.load_dataset("tips")
df.head()


# # Correlation with Heatmap
# 
# A correlation heatmap uses colered cells, typically in a monochrome scale, to show a 2D correlation matrix (table) between two descrete dimensions or event types. It is very important in feature Selection.

# In[5]:


df.corr()


# In[7]:


sns.heatmap(df.corr())


# ## Join plot
# A Join Plot allows to study the relationship between 2 numeric variables. The central chart display their correlation it ususlly a scatterplt, a hexbin plot, 2D histogram or a 2D density plot.
# 
# Bivariate analysis

# In[158]:


sns.jointplot(x='tip',y='total_bill',data=df,kind='hex')


# In[155]:


sns.jointplot(x='tip',y='total_bill',data=df)


# In[160]:


sns.jointplot(x='tip',y='total_bill',data=df,kind='reg')


# # pair Plot 
# A pair plot is also known as a scatterplot, in which one variable in the same data row is matched with another variables value, like thsi plots are just elaborations on this, showing all variable 

# In[152]:


sns.pairplot(data=df)


# In[23]:


sns.pairplot(df,hue='sex')


# # Dist Plot
# Dist plot helps us to check the distribution of the colums feature
# 
# Note:if kd is True we will get in percentage.

# In[154]:


sns.distplot(df['tip'])


# In[15]:


sns.displot(df['tip'],kde=True,bins=10)


# ## Categorical plots
# Seaborn also helps us in doing the analysis on Categorical Data points. In this we will discuss about 
# * Box plot
# * Violin plot
# * Bar plot

# In[20]:


df


# ### Count Plot

# In[25]:



sns.countplot('day',data=df)


# In[61]:


sns.countplot(x='sex',data=df)


# ### Bar plot

# In[66]:


sns.barplot(x='total_bill',y='sex',data=df)


# In[65]:


#Note: In Barplot you have to specially give x and y value 


# ### Box Plot
# A box and whisker plot (sometimes called a boxplot) is a graph that present information from a five_number summary 

# In[67]:


sns.boxplot(x='smoker',y='total_bill',data=df)


# In[69]:


#Note: We can also add colors in boxplot which is denoted by palette='rainbow'


# In[72]:


sns.boxplot(x='day',y='total_bill',data=df,palette='rainbow',orient='v')


# In[76]:


sns.boxplot(data=df,orient='h')


# In[77]:


#Categorize my data based on some other categores


# In[78]:


sns.boxplot(x='total_bill',y='day',hue='smoker',data=df)


# ### Violin Plot
# Violin plot helps us to see both the distribution of data in terms of kernel density estimation and box plot.

# In[79]:


sns.violinplot(x='total_bill',y='day',data=df,palette='rainbow')


# In[140]:


sns.violinplot(x='total_bill',y='smoker',data=df,palette='rainbow')


# In[131]:


iris=sns.load_dataset('iris')


# In[ ]:




