## Loading and Saving Data: Pandas provide several ways to load and save data into and from a DataFrame:


import pandas as pd
import sqlite3
import numpy as np

# Loading Data from a CSV File: You can load data from a CSV file using the “read_csv” function. 
df = pd.read_csv('data.csv') 


# Loading Data from an Excel File: You can load data from an Excel file using the “read_excel” function. 
df = pd.read_excel('data.xlsx') 


# Loading Data from a SQL Database: You can load data from a SQL database using the “read_sql” function. 
conn = sqlite3.connect("database.db")
df = pd.read_sql("SELECT * FROM table_name", conn)


# Loading Data from a Dictionary: You can load data from a dictionary using the DataFrame constructor.
data = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data) 


# Saving Data to a CSV File: You can save data to a CSV file using the “to_csv” method. 
# …all df manipulation
df.to_csv('data.csv', index=False) 


#Saving Data to an Excel File: You can save data to an Excel file using the “to_excel” method.
# …all df manipulation
df.to_excel('data.xlsx', index=False)


#Saving Data to a SQL Database: You can save data to a SQL database using the “to_sql” method
conn = sqlite3.connect('database.db')
df = pd.read_sql("SELECT * FROM table_name", conn)
# …all df manipulation
df.to_sql('table', conn, if_exists='replace')



#########################################################

# Creating a Series from a list
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)


# Output:
# 0    1.0
# 1    3.0
# 2    5.0
# 3    NaN
# 4    6.0
# 5    8.0
# dtype: float64


# Creating a Series from a dictionary
s = pd.Series({'a': 1, 'b': 3, 'c': 5})
print(s)


# Output:
# a    1
# b    3
# c    5
# dtype: int64


# Extracting a column from a DataFrame
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
s = df['col1']
print(s)


# Output:
# 0    1
# 1    2
# Name: col1, dtype: int64


# Indexing using Square Brackets: You can select a single column of a DataFrame by using square brackets and the column name.
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
print(df['col1'])


#Indexing using Dot Notation: You can also access a column using dot notation, but this method is not recommended because it can lead to confusion with the methods and attributes of the DataFrame.
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
print(df.col1)


# Indexing using Row/Column Labels: You can select rows of a DataFrame using row/column labels using the “loc” attribute. 
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
df.index = ['row1', 'row2']
print(df.loc['row1','col1'])


#Indexing using Integer-Based Location: You can select rows/columns of a DataFrame using integer-based location using the “iloc” attribute.
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
print(df.iloc[0:2, 1])


# output
# 0    3
# 1    4
# Name: col2, dtype: int64


#Boolean Indexing: You can select rows of a DataFrame based on a condition using boolean indexing.
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
print(df[df['col1'] > 1])


#################################################################

# Creating a sample dataframe with missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [6, np.nan, 8, 9, 10],
    'C': [11, 12, 13, 14, np.nan]
})


# Display the dataframe
print("Original Dataframe:")
print(df)


# Output
# Original Dataframe:
#       A     B     C
# 0  1.0   6.0  11.0
# 1  2.0   NaN  12.0
# 2  NaN   8.0  13.0
# 3  4.0   9.0  14.0
# 4  5.0  10.0   NaN



# Identifying the missing values
print("\nMissing values:")
print(df.isna().sum())
# Output
# Missing values:
# A    1
# B    1
# C    1
# dtype: int64



# Filling the missing values with a constant value (0)
df_fillna = df.fillna(0)
print("\nDataframe after filling missing values with 0:")
print(df_fillna)
# Output
# Dataframe after filling missing values with 0:
#       A     B     C
# 0  1.0   6.0  11.0
# 1  2.0   0.0  12.0
# 2  0.0   8.0  13.0
# 3  4.0   9.0  14.0
# 4  5.0  10.0   0.0



# Dropping the rows containing missing values
df_dropna = df.dropna()
print("\nDataframe after dropping rows with missing values:")
print(df_dropna)
# Output
# Dataframe after dropping rows with missing values:
#       A    B     C
# 0  1.0  6.0  11.0
# 3  4.0  9.0  14.0



# Dropping the columns containing missing values
df_dropna_col = df.dropna(axis=1)
print("\nDataframe after dropping columns with missing values:")
print(df_dropna_col)
# Output
# Dataframe after dropping columns with missing values:
# Empty DataFrame
# Columns: []
# Index: [0, 1, 2, 3, 4]


##################################################

# Create a sample dataframe
df = pd.DataFrame({'A':[1,2,3,4,5,6,7,8,1,2,3],
                   'B':['a','b','c','d','e','f','g','h','a','b','c']})


# Check the original dataframe
print("Original dataframe:")
print(df)
# Output
# Original dataframe:
#     A  B
# 0   1  a
# 1   2  b
# 2   3  c
# 3   4  d
# 4   5  e
# 5   6  f
# 6   7  g
# 7   8  h
# 8   1  a
# 9   2  b
# 10  3  c


# Identify the duplicate rows
duplicates = df.duplicated()
print("Duplicate rows:")
print(duplicates)
# Output
# Duplicate rows:
# 0     False
# 1     False
# 2     False
# 3     False
# 4     False
# 5     False
# 6     False
# 7     False
# 8      True
# 9      True
# 10     True
# dtype: bool


# Remove the duplicate rows
df = df.drop_duplicates()


# Check the data frame after removing duplicates
print("Dataframe after removing duplicates:")
print(df)
# Output
# Dataframe after removing duplicates:
#    A  B
# 0  1  a
# 1  2  b
# 2  3  c
# 3  4  d
# 4  5  e
# 5  6  f
# 6  7  g

#################################################

# Create a sample dataframe
df = pd.DataFrame({'A':[1,2,3,4,5,6,7,8,1,2,3],
                   'B':['a','b','c','d','e','f','g','h','a','b','c']})


# Check the original dataframe
print("Original dataframe:")
print(df)
# Output
# Original dataframe:
#     A  B
# 0   1  a
# 1   2  b
# 2   3  c
# 3   4  d
# 4   5  e
# 5   6  f
# 6   7  g
# 7   8  h
# 8   1  a
# 9   2  b
# 10  3  c


# Identify the duplicate rows
duplicates = df.duplicated()
print("Duplicate rows:")
print(duplicates)
# Output
# Duplicate rows:
# 0     False
# 1     False
# 2     False
# 3     False
# 4     False
# 5     False
# 6     False
# 7     False
# 8      True
# 9      True
# 10     True
# dtype: bool


# Remove the duplicate rows
df = df.drop_duplicates()


# Check the data frame after removing duplicates
print("Dataframe after removing duplicates:")
print(df)
# Output
# Dataframe after removing duplicates:
#    A  B
# 0  1  a
# 1  2  b
# 2  3  c
# 3  4  d
# 4  5  e
# 5  6  f
# 6  7  g

################################################


# Creating two simple dataframes
df1 = pd.DataFrame({
   'A': ['A0', 'A1', 'A2', 'A3'],
   'B': ['B0', 'B1', 'B2', 'B3'],
   'key': ['K0', 'K1', 'K2', 'K3']
})


df2 = pd.DataFrame({
   'C': ['C0', 'C1', 'C2', 'C3'],
   'D': ['D0', 'D1', 'D2', 'D3'],
   'key': ['K0', 'K1', 'K2', 'K4']
})


print("df1:")
print(df1)
print("\ndf2:")
print(df2)


# Concatenation: It is the simplest method of combining data in pandas. It is performed using the “pd.concat()” method. This method takes a list of data frames as input and concatenates them along the resulting data frame's rows (axis=0) or columns (axis=1).
concat = pd.concat([df1, df2], axis=0)
print("\nConcatenated dataframe:")
print(concat)


# Inner Join: It is the default join in pandas. Only the common rows are kept in the resulting data frame in this type of join. All the other rows are discarded.
inner_join = pd.merge(df1, df2, on='key', how='inner')
print("\nInner Join:")
print(inner_join)


# Left Join: In a left join, all the values from the left dataframe are combined with those from the right dataframe. The resulting dataframe includes all the values from the left dataframe and the matching values from the right dataframe. For the missing values in the right dataframe, NaN values are used.
left_join = pd.merge(df1, df2, on='key', how='left')
print("\nLeft Join:")
print(left_join)


# Right Join: In a right join, all the values from the right dataframe are combined with those from the left dataframe. The resulting dataframe includes all the values from the right dataframe and the matching values from the left dataframe. For the missing values in the left dataframe, NaN values are used.
right_join = pd.merge(df1, df2, on='key', how='right')
print("\nRight Join:")
print(right_join)


# Outer Join: In an outer join, all the values from both dataframes are combined, and the resulting dataframe includes all the values from both dataframes, even if a value is not present in one of the dataframes
outer_join = pd.merge(df1, df2, on='key', how='outer')
print("\nOuter Join:")
print(outer_join)

####################################################

import pandas as pd


# Create a sample dataframe
df = pd.DataFrame({
   'A': ['foo', 'bar', 'baz', 'foo', 'bar', 'baz'],
   'B': ['one', 'one', 'two', 'two', 'one', 'one'],
   'C': [1, 2, 3, 4, 5, 6],
   'D': [10, 20, 30, 40, 50, 60]
})


# Encoding categorical variables
df['A_encoded'] = df['A'].astype('category').cat.codes
df['B_encoded'] = df['B'].astype('category').cat.codes


# Normalization - Min-Max scaling (range 0 to 1)
df['C_norm'] = (df['C'] - df['C'].min()) / (df['C'].max() - df['C'].min())
df['D_norm'] = (df['D'] - df['D'].min()) / (df['D'].max() - df['D'].min())


# Advanced groupby operation - group by 'A' and 'B', calculate mean of 'C_norm' and sum of 'D_norm'
grouped_df = df.groupby(['A', 'B']).agg({'C_norm':'mean', 'D_norm':'sum'}).reset_index()


print(grouped_df)