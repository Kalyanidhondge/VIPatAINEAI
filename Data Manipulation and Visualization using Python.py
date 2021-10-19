#!/usr/bin/env python
# coding: utf-8

# In[1]:


# TO DO: Import pandas library as an alias name (nickname) pd
# Syntax hint: import libraryname as aliasname
import pandas as pd


# In[2]:


# Getting data from local location in our system.
brics = pd.read_csv(r'C:\Users\dkk\AINE AI - Intern\project-6-Data-Manipulation-and-Visualization\brics.csv')  # Change this path as per your folder arrangements.

# Make sure the folder names are seperated by either a forward slash (/)or two backward slashes (\\) in the above code.

brics


# In[3]:


# Data type of any variable can be seen using the function type()

print(type(5))
print(type('a text'))
print(type(True))
print(type(brics))         # datatype of the dataframe created above


# In[4]:


'''
The info() function is used to print a concise summary of a DataFrame. 
This method prints information about a DataFrame including the index dtype and column dtypes, 
non-null values and memory usage.
'''
brics.info()

# Object dtype can store variables with any or mixed data types


# In[5]:


#value_counts() function returns object containing counts of unique values. 
#The resulting object will be in descending order so that the first element is the most frequently-occurring element. 
#Excludes NA values by default.

# TO DO: use value_counts() in the brics dataframe to explore the data.
brics = pd.Series(brics)
brics.value_counts()


# In[6]:


# When the dataset is large and you want to sort the data by ascending or descending order to look at largest or smallest values.

# Ascending order by column code. Uncomment the lines to run them.
# brics.sort_values(by=['code'])   # Ascending by default.
# or
# brics.sort_values(by=['code'], ascending = True)  # No need to mention
# Sorting in ascending order.
brics.sort_values(by=['code'])


# In[7]:


# Sorting in descending order.
brics.sort_values(by=['code'], ascending = False)


# In[8]:


# TO DO: Do you think sorting the data changes the data in original dataframe? 
# print the dataframe and observe the output to get the answer.
brics


# In[10]:


# No I don't think sorting the data changes the data in original dataframe.
# TO DO: Sort the data to see which country has largest population ( population is in millions).
brics.sort_values(by=['population'],ascending = False)
# From output it is clear that China has largest population


# In[11]:


# TO DO: Sort the data to see which country is smallest in size.
brics.sort_values(by=['area'])
# From below output it is clear that South Africa is smallest in size.


# In[12]:


# Sorting and printing just one column.

brics.country.sort_values(ascending=False)


# In[13]:


# TO DO: Sort and print area column on from largest to smallest.

brics.area.sort_values(ascending = False)


# In[14]:


# Indices can be used to fetch data from a particular row. 
# Example, to fetch data from the 2nd row, we can use index 1 (one)
brics.country[1]

# We will work more on the index column later.


# In[15]:


# Getting single column from a dataframe

print(brics["country"])


# In[16]:


# TO DO: Print area column from brics dataframe
print(brics['area'])


# In[17]:


# Other way to get a single column from a dataframe - use dot

brics.country


# In[18]:


# TO DO:  Print code column from the dataframe using dot
brics.code


# In[19]:


# Pritning multiple columns
brics[['country', 'capital']]


# In[20]:


# TO DO: Print country, capital and population

brics[['country','capital','population']]


# In[21]:


df = pd.DataFrame({'name': ['Niko', 'Penelope', 'Aria'], 'average score': [10, 5, 3], 'max': [99, 100, 3]})

df['average score']


# In[23]:


# use col 0 as index. 
# This will remove the 1st column containing the index or sl. no. from the above output, 
# and make the 1st column from actual data as the index.

brics = pd.read_csv(r'C:\Users\dkk\AINE AI - Intern\project-6-Data-Manipulation-and-Visualization\brics.csv',index_col=0)
brics


# Note that the 'code' will be used as Index only and not as a part of data anymore.


# In[24]:


#Adding a new column to the existing Data Frame
brics["on_earth"] = [True, True, True, True, True]    

# We are adding a list here. Read more on lists: https://www.w3schools.com/python/python_lists.asp

brics


# In[25]:


# adding a calculated column
brics["density"] = brics["population"] / brics["area"] * 1000000   # per km sq.
brics


# In[26]:


brics.style.set_precision(2)  # Upto 2 decimal points for all float columns in the dataframe. 

# This WILL NOT change the original dataframe i.e. the original dataframe will still have multiple digits after decimals.


# In[27]:


brics["density"].round(2) # Rounding of individual column to fixed decimal points.


# In[28]:


# Before starting, let's first print the brics daraframe. Can you please do this?
brics
# Notice that the indices are BR, RU, IN, CH and SA. However, their position (numerical index) are 0, 1, 2, 3 and 4.


# In[29]:


brics.loc["SA"]     # .loc is used with actual value of the index. 
# The value mentioned in the argument must be there in the dataframe's index list.


# In[30]:


brics.iloc[4]  # .loc is used when we are giving the index reference as number, not the actual value of the index.


# In[31]:


# TO DO: Use .loc to print the values from all columns for IN
brics.loc['IN']


# In[32]:


# TO DO: Use .iloc to print the values from all columns from 3rd row (index number will be 1 less than the row number).
brics.iloc[2]


# In[33]:


#ading a new ROW using append
newrow = {'code':'WK', 'country':'Wakanda', 'population':5,'area':1000000,'capital':'Wakanda City','on_earth':False,'density':5}
brics1 = brics.append(newrow,ignore_index=True)   # Ignoring the existing index.
# Ignoring the existing index will result into new dataframe index 0,1,2,....

'''
Why there is NaN for code in other rows?  
Because the previous 'code' was not a part of datatable. It was an index.
With 'code' in 'newrow', we are adding a new column called 'code'
'''

'''
ignore_index=False will result into error because 'newrow' is a dictionary
and appending a dictionary into a dataframe will not work if we do not ignore the index.
'''
brics1   # append will not result into chaning the actual dataframe. brics will still be the same


# In[34]:


#Adding a new row using .loc
brics.loc['WK'] = ['Wakanda', 5, 1000000, 'Wakanda City', False, 5]   
# Sequence is very important. Data types may get changed due to wrong sequence.
brics


# In[35]:


# TO DO: Add a new row in the dataframe using .loc any other country with some values in all columns.
brics.loc['ENG'] = ['England',1756,5600000,'Londan',True,6]
brics


# In[36]:


# Fetching the value in 2-D way. In Excel, we have cells for that, Ex- A2 means column A and row 2.
# All of the below codes will give the same output. Please uncomment and execute them one by one.

brics.loc["IN","capital"]


# In[37]:


brics["capital"].loc["IN"]


# In[38]:


brics.loc["IN"]['capital']


# In[39]:


# Lets create a new dataframe so that brics is not affected.

brics1 = brics
brics1


# In[40]:


brics1.drop(['area'], axis=1)  # axis = 0 by default for rows. Axis = 1 for column.

# For multiple columns, mention the columns separated by comma. Ex. df.drop(['name','max'], axis=1)


# In[41]:


brics1 = brics
# TO DO: Try deleting area and density columns from brics1 dataframe
brics1.drop(['area','density'],axis = 1)


# In[42]:


# TO DO: Delete the record for Wakanda by using its index WK
# Hint: Use WK as identifier and delete axis argument or put axis = 0
brics1=brics
brics1.drop('WK',axis=0)


# In[43]:


brics1 = brics

# Another way to drop columns
brics1.drop(columns=['area', 'density'])


# In[44]:


brics1 = brics

# Drop rows by index

brics1.drop(['IN', 'CH'])


# In[45]:


# The shape function helps us to find the shape or size of an array or matrix. In Excel - A1:D10.  
# shape[0] means we are working along the first dimension of your array.
# If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n.

for index in range(brics.shape[0]):    
    countryName = brics.iloc[index,0]  # row - index, column - 0
    cityName = brics.iloc[index, 3]  # row - index, column - 3
    print('The Capital City of', countryName, 'is', cityName)


# In[46]:


#Another solution
for index in range(brics.shape[0]):
    print('The Capital City of', brics.iloc[index, 0], 'is', brics.iloc[index, 3])


# In[47]:


#Another solution
# iterrows() is a generator that iterates over the rows of the dataframe and returns the index of each row, 
# in addition to an object containing the row itself.

for index, row in brics.iterrows():
    print("The Capital City of",row['country'],"is", row['capital'])


# In[48]:


cars = pd.read_csv(r'C:\Users\dkk\AINE AI - Intern\project-6-Data-Manipulation-and-Visualization\cars.csv')

# Print top 5 rows from the data, without any sorting or filter.
cars.head()


# In[49]:


# TO DO: Print top 7 rows from the data
# To print n number of rows, enter the value of n in the head() as an argument.

cars.head(7)


# In[50]:


# For bottom 5 rows in the dataframe
cars.tail(5)


# In[51]:


# TO DO: Print last 10 rows from the dataframe
cars.tail(10)


# In[52]:


# Giving a name to the unnamed column

print(cars)

df1 = cars.rename( columns={'Unnamed: 0':'code'}, inplace=False ) 
cars.rename( columns={'Unnamed: 0':'code'}, inplace=True ) 

print(df1)
print(cars)      # With False, there will be no change in cars dataframe.



'''
inplace = True has been used to overwrite the existing dataframe. The default is False, if nothing mentioned.
When inplace = True , the data is modified in place, which means it will return nothing 
and the dataframe is now updated. 
When inplace = False , which is the default, 
then the operation is performed and it returns a copy of the object. You then need to save it to something.'''


# In[53]:


print(cars['cars_per_cap'])  # without column name at the top. Prints data and detail


# In[54]:


print(cars[['cars_per_cap']])  # print the actual data


# In[55]:


# TO DO: Read the data again and set the 1st column as the index column. Then, print the cars dataframe.
cars = pd.read_csv(r'C:\Users\dkk\AINE AI - Intern\project-6-Data-Manipulation-and-Visualization\cars.csv',index_col=0)
cars


# In[56]:


# TO DO : Print the details of RU
# Hint: You should use either .loc or .iloc (just one is correct).
cars.loc['RU']


# In[57]:


# TO DO: Can you change the above answer to show the column name at the top? 
# Hint: Recap last 3-4 code cells.
cars.loc[['RU']]


# In[58]:


# TO DO: Print all details for RU and AUS from cars dataframe.
cars.loc[['RU','AUS']]


# In[59]:


# Print the Cars Per Capita (cars_per_cap) for India (IN).
cars.loc['IN']['cars_per_cap']


# In[60]:


# Print the Cars Per Capita (cars_per_cap) for India (IN) and Russia (RU).
cars.loc[['IN','RU']]['cars_per_cap']


# In[61]:


# Print the Cars Per Capita and Country name for India and Russia
cars.loc[['IN','RU']][['cars_per_cap','country']]


# In[62]:


# If working on Jupyter Notebook and data is in local drive
marks = pd.read_csv(r'C:\Users\dkk\AINE AI - Intern\project-6-Data-Manipulation-and-Visualization\marks.csv')
marks


# In[63]:


# TO DO: Show the shape of the dataframe
# Hint: shape()
marks.shape


# In[64]:


# TO DO: Get the information and structure of the data with non-null value counts
marks.info()


# In[65]:


# Print datatypes of the each column
print(type(marks.Student_ID))
print(type(marks.Student_Name))
print(type(marks.English))
print(type(marks.Maths))
print(type(marks.Science))
print(type(marks.History))
print(type(marks.Social_Studies))


# In[66]:


print(type(marks))
print(type(marks.English))  # Column's dtype is Series, the dtype of the values is Object.
print(type(5))
print(type(5.5))
print(type("Python"))
print(type(True))


# In[67]:


# Run these lines one by one and observe the difference in the three outputs.
print(marks)


# In[68]:


display(marks)


# In[69]:


marks


# In[70]:


num = 5 # Assignment. No output of assignment operator. 
num

num == 6 # Comparison/equality. Output is either True or False


# In[71]:


# selecting marks for 'Ria'
display(marks.loc[(marks.Student_Name == 'Ria')])  # in case the index is not known for large datasets


# In[72]:


# TO DO: Print all records for David
display(marks.loc[(marks.Student_Name == 'David')])


# In[73]:


# Using OR operator to print details if the name is either 'Ria' OR 'David'
display(marks.loc[(marks.Student_Name == 'Ria') | (marks.Student_Name == 'David')])


# In[74]:


# TO DO: Print all records is the Student IS is either S05 or S10
display(marks.loc[(marks.Student_ID == 'S05') | (marks.Student_ID == 'S10')])


# In[75]:


# TO DO: Fetch all details where marks in English > 70 or marks in Maths > 70.
display(marks.loc[(marks.English > 70) | (marks.Maths>70)])


# In[76]:


# Using AND operator to combine two conditions. 

# TO DO: Fetch all details where marks in English > 70 AND marks in Maths > 70.
# Hint: Use & for AND
display(marks.loc[(marks.English > 70) & (marks.Maths>70)])


# In[77]:


# Using NOT function 

# TO DO: Fetch all details for all students except for Ria or the all students who are not Ria.
# Hint: Use != for 'not equal to'
display(marks.loc[(marks.Student_Name != 'Ria')])


# In[79]:


# Index column can be changed by giving the name of the column - unique and case-sensitive
marks = pd.read_csv(r'C:\Users\dkk\AINE AI - Intern\project-6-Data-Manipulation-and-Visualization\marks.csv',index_col = 'Student_ID')  # could have given the index no. of the columns (0,1..) as well.
display(marks)


# In[80]:


display(marks.loc[1:10]) # Range of values. Includes all values/labels.

# .loc is different than .iloc
# .loc is label-based. We need to SPECIFY the VALUE of the index. The values must be there in the index column.


# In[81]:


'''
iloc is integer index-based. 
So here, we have to specify rows and columns by their integer index.
'''
display(marks.iloc[0:10]) # IMPORTANT - Out of 11 indices, the last one will be excluded.

# Returns output of the indices without matching the value of incides entered and the value in the column.
# iloc will not work if the indices are not integers.


# In[82]:


# What will be the output here? First think of the answer and then run the code.

display(marks.iloc['S04'])

# What is the issue?
#Ans - iloc is the integer index base, hence cannot index by location index with a non-integer key


# In[83]:


# Correct the issue of the previous code cell and write the correct syntax here.
display(marks.loc['S04'])


# In[84]:


# Data of multiple and non-consecuitive indices
# Selecting details of the 1st and 5th row
display(marks.iloc[[0, 4]])


# In[85]:


# TO DO: Print the details from 0th, 5th and 8th rows, along with the details of indices 4th, 11th and 16th.
display(marks.iloc[[0,4,8,3,10,15]])


# In[86]:


# .loc can be used to fetch details from a range of index values in a sequence.

display(marks.loc['S01':'S05'])   # Continuous series of the indices based on label.


# In[87]:


# Run these line one by one and observe the output to notice the difference and execution of the syntax
display(marks.iloc[2:5])


# In[88]:


display(marks.iloc[0:5])


# In[89]:


display(marks.iloc[:5])


# In[90]:


display(marks.iloc[15:20])


# In[91]:


display(marks.iloc[15:])


# In[92]:


display(marks.iloc[15:100])


# In[93]:


display(marks.iloc[:])


# In[94]:


# TO DO: Copy paste the above codes and explain the output for each syntax as comments as per given example.
# Example: display(marks.iloc[[3:5]])  # Prints all values from all columns for indices 2, 3 and 4.
# display(marks.iloc[2:5]) prints all values from all columns for indices 0,1,2,3 and 4.
# display(marks.iloc[0:5]) prints all values from all columns for indices from 0 to 4.
# display(marks.iloc[:5])   prints all values from all columns for indices from 0 to 4.
# display(marks.iloc[15:20])  prints all values from all columns for indices from 14 to 19.
# display(marks.iloc[15:]) prints all values from all columns for indices from 14 to end.
# display(marks.iloc[15:100]) prints all values from all columns for indices from 14 to 99.
# display(marks.iloc[:])  prints all values from all columns for indices from start to end.
#.................... Data cleaning : Empty value treatment iin dataframes

temps = pd.DataFrame({"sequence":[1,2,3,4,5],
          "measurement_type":['actual','actual','actual',None,'estimated'],  # With strings, it will become 'None'
          "temperature_f":[67.24,84.56,91.61,None,49.64]   #With numbers, it will become 'NaN'
         })
temps


# In[95]:


# To identify the null value. Return true if the value is null otherwise false
temps.isna()


# In[96]:


# Return the count of missing value from each column     
temps.isna().sum()


# In[97]:


# Total count of null from all columns
temps.isna().sum().sum()


# In[98]:


# Drop/delete the row containing missing values

clean_temps = temps.dropna(how='any')  # how: {'any', 'all'}. Default 'any'
display (clean_temps)

# ‘any’ : If any NA values are present, drop that row or column.
# ‘all’ : If all values are NA, drop that row or column.


# Observe, in the output, index 3 is not there.


# In[99]:


# TO DO:  Copy paste the codes from the above cell, change the 'how' argument as 'all' and observe the difference in output.
clean_temps = temps.dropna(how='all') 
display (clean_temps)


# In[100]:


# Drop the ROWS where at least one element is missing.
temps.dropna()


# In[101]:


# Drop COLUMNS where there are missing values.

# Drop the columns where at least one element is missing.
temps.dropna(axis = 'columns') # Without the 'axis' argument, rows will be dropped by default as you did in the previous code.


# In[102]:


temps['temperature_f'].cumsum()   # Returns the commulative sum. 

# It will skip null values. skipna = TRUE by default

# CAN YOU THINK OF ANY PRACTICAL USE OF cumsum() ?
# Ans - used to calculate the cumulative sum of the vector.


# In[103]:


temps['temperature_f'].cumsum(skipna=False)


# In[104]:


# fill missing value with zero
temps.fillna(value=0, inplace = True)   # Do you remember the use of inplace? If not, please scroll up and check.
display(temps)


# In[105]:


# Print the dataframe

temps

# Why 0 in last column is 0.00 while only 0 in measurement_type? Please explain as a comment in this cell.
# Ans = Data type of measurement_type column is 'str' and data type of column temperature_f is 'float', hence 0 in last column 
# is 0.00 while only 0 in measurement_type.


# In[106]:


# fill missing value with previous value
temps = pd.DataFrame({"sequence":[1,2,3,4,5],
          "measurement_type":['actual','actual','actual',None,'estimated'],
          "temperature_f":[67.24,84.56,91.61,None,49.64]
         })
temps.fillna(method='pad' , inplace=True)  # 'pad' means padding. Take value from previous row
temps


# In[107]:


# fill missing value with next value
temps1 = pd.DataFrame({"sequence":[1,2,3,4,5],
          "measurement_type":['actual','actual','actual',None,'estimated'],
          "temperature_f":[67.24,84.56,91.61,None,49.64]
         })
temps1.fillna(method='bfill' , inplace=True)  # bfill takes next value to replace
temps1


# In[108]:


# fill missing value with mean
temps = pd.DataFrame({"sequence":[1,2,3,4,5],
          "measurement_type":['actual','actual','actual',None,'estimated'],
          "temperature_f":[67.24,84.56,91.61,None,49.64]
         })
temps['temperature_f'].fillna(temps['temperature_f'].mean(), inplace=True)  # Mean will not work on strings
temps


# In[109]:


# lambda is used to define a temporary expression without any return statement. It always contains an expression
# that is returned. There is no need to assign a variable with lambda.


def cube(y):
    return y*y*y  # return is a keyword. Python stops when the code reaches to return statement. Print is a function.
    

# using the normally defined function
    print(cube(5))
 
# using the lambda function    # Why this is red? How to resolve it?
# Ans - Above line is red due to indentation and it can be resolve by select the above line and press "Tab" 
lambda_cube = lambda y: y*y*y
print(lambda_cube(5))


# In[110]:


# Creating new dataframe to explore the use of Lambda and changing indices.

teams = pd.DataFrame({"Region":['North','West','East','South'],
          "Team":['One','Two','One','Two'],
          "Squad":['A','B','C','D'],
          "Revenue":[7500,5500,2750,6400],
            "Cost":[5200,5100,4400,5300]})

display (teams)


# In[111]:


# apply() to alter values along an axis in your dataframe or in a series/column 

# Categorise based on the revenue and cost
teams['Profit'] = teams.apply(lambda x: 'Profit' if x['Revenue']>x['Cost'] else 'Loss',axis=1)
teams


# In[112]:


# Use map() to substitute each value in a series
team_map = {"One":"Red","Two":"Blue"}      # new variable - dictionary (key-value pair)
teams['Team Color'] = teams['Team'].map(team_map) # A new column with mapped values
teams


# In[113]:


# applymap() method do elementwise operation on the entire dataframe.
# This method applies a function that accepts and returns a scalar to every element of a DataFrame.

teams.applymap(lambda x: len(str(x)))  # int(x) won't work because data has strings which can not be convereted to int.


# In[114]:



# Grouping on different categories. Needs the category as well as parameter
teams.groupby(['Profit']).max()


# In[115]:


# TO DO: Can you group the records on Cost with minimum values?
teams.groupby(['Cost']).min()


# In[116]:


# TO DO: Can you group the records on Revenue column with mean values?
teams.groupby(['Revenue']).mean()


# In[117]:


# Grouping on the basis of aggregates
teams.groupby(['Team']).agg({'Revenue':['mean','min','max']})


# In[118]:


# TO DO: Copy-paste the copy from earlier to redefine teams dataframe, and display it
teams


# In[119]:


# Setting two columns as index when a single column has not unique values.

teams_reindex = teams.set_index(['Region','Team'])  # 2 indices. useful when values are not unique
display(teams)

print()    # TO print blank/new line.


display(teams_reindex)


# In[120]:


# Restructuring the dataframe based the multiple indices
stacked = pd.DataFrame(teams_reindex.stack())
stacked


# In[123]:


# Mergining dataframes
# We are going to learn how to merge multiple dataframes based on left, right and inner join.

# Defining two new dataframes
group1 = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'David'],
                    'number': [1, 2, 3, 4]})
group2 = pd.DataFrame({'name': ['Charlie', 'David', 'Edward', 'Ford'],
                    'number': [3, 4, 5, 6]})

group1.merge(group2,how='left', on='number')   # Left - all from 1st table and only common from 2nd table.

# Notice the NAN or empty values.


# In[124]:


group1.merge(group2)   # Shows only the common records.


# In[125]:


group1.merge(group2,how='inner',left_on='number',right_on='number')  # Inner - Shows only the common records

# Why there are no null values now in the output? Answer as a comment.
# There are no null values in output because inner join shows only common records.


# In[126]:


group1.merge(group2,how='right',left_on='number',right_on='number')  # Right - All from 2nd table, common from 1st table.


# In[127]:


# TO DO : Read the instructions and write the codes.
'''
Get the data

Open mtcars.csv in Excel or Text dataset and have a look at the data. Understand the variables.

You can also open in Google drive to view it however, make sure you download it in your local system and upload in 
Jupyter Notebook folder where you have the Python file.
'''
mtcars = pd.read_csv(r'C:\Users\dkk\AINE AI - Intern\project-6-Data-Manipulation-and-Visualization\mtcars.csv')
mtcars


# In[128]:


# TO DO: 
# Import pandas package as pd
import pandas as pd
# Use function pd.read_csv to upload mtcars.csv by using either of two methods below:
    # Use_cols ( all the columns except car_names )
    # Index_col = car_names ( expected output as below)
mtcars = pd.read_csv(r'C:\Users\dkk\AINE AI - Intern\project-6-Data-Manipulation-and-Visualization\mtcars.csv', index_col = 'car_names')
mtcars
# display the dataframe


# In[129]:


# Import he required package 
import matplotlib.pyplot as plt

# TO DO : Create a scatter plot with mpg and hp by providing xlabel and ylabel
plt.scatter(mtcars.mpg,mtcars.hp)
plt.xlabel('mpg')
plt.ylabel('hp')
plt.title('Scatter Plot mpg VS hp')
plt.show()


# In[131]:


# TO DO: Change the color to red in the previous code
plt.scatter(mtcars.mpg,mtcars.hp,color = 'red')
plt.xlabel('mpg')
plt.ylabel('hp')
plt.title('Scatter Plot mpg VS hp')
plt.show()


# In[132]:


# TO DO: Create crosstab for cyl(cylinder) and columns as count

# A cross-tabulation (or crosstab) is a two- (or more) dimensional table that records the number (frequency) of 
# respondents that have the specific characteristics described in the cells of the table.
pd.crosstab(mtcars.cyl,mtcars.cyl.count())


# In[133]:


# Other way to create the crosstab
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ct = pd.crosstab(mtcars.cyl, mtcars.cyl)

ct.plot.bar(stacked=True)    # Try staked=False and then by removing stakced part
plt.legend(title='mark')   # Optional

plt.show()


# In[134]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ct = pd.crosstab(mtcars.cyl, mtcars.cyl)

ct.plot.bar(stacked=False)    # Try staked=False and then by removing stakced part
plt.legend(title='mark')   # Optional

plt.show()


# In[135]:


# TO DO: Create crosstab for cyl(cylinder) and am as 'ct2' and create a bar plot

# Reading reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
pd.crosstab(mtcars.cyl,mtcars.am)


# In[136]:


ct = pd.crosstab(mtcars.cyl, mtcars.am)

ct.plot.bar(stacked=False)    # Try staked=False and then by removing stakced part
plt.legend(title='mark')   # Optional

plt.show()


# In[137]:


# TO DO: Create groupby (cyl) to visualize mpg and hp in a bar chart
# Hint: Use nunique() 
ct1 = mtcars.groupby(['cyl'])
plt.bar(mtcars.mpg,mtcars.hp,color='blue')
plt.xlabel('mpg')
plt.ylabel('hp')
plt.show()


# In[141]:


# TO DO: Create a Histogram for MPG column
# Reading: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.hist.html
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(mtcars.mpg)
plt.show()


# In[ ]:




