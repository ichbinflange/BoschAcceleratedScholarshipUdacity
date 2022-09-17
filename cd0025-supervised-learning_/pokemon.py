
import pandas as pd
import re

df = pd.read_csv('pokemon_data.csv')
print (df.head(4))
#pd.read_excel
#i want to find the headers
print(df.columns)
#to read a specific column like $ in r putting the number of the row in []
print(df.Name[5])
#printing out each row
print(df.head(4))
# i want to print out index row we have to use the iloc which means initial location
print(df.iloc[4:10])
# : is used to get the sequence of values while the , will give me a specific location
print(df.iloc[2,1])
#to go row by row and access any sort of data i want 

for index,row in df.iterrows():
    print(index,row['Name'])
    
#df.loc and can access only rows that have 'type 1 = fire
# it display the entire row according to a specific column property
    print(df.loc[df['Type 1']== 'Fire'])
    
 #this is lke the summary in R
df.describe()
#sort the values alphabetically
df.sort_values('Name',ascending = False)

#we can also make the sorting more limited by choosing the levels
#werll the most important thing is to have the  syntax 
#so sorting according to type 1,type 2 once is ascending 0 is desending
newsort = df.sort_values(['Type 1','HP'], ascending = [1,0])

#so lets start making change the data
#open a new column 
df['total'] = df.iloc[:,4:10].sum(axis = 1)
 #remember : means select all columns in this case 
 #axis = 1 is  horizontally 0 is vertically
 #when performing operations using columns always add the last next 
#dropping or removing a column
#df = df.drop(columns = ['total']) it just removed the total column or dropped
#we can actualy move our column position and grab it to another place which is reorder

cols = list(df.columns.values)
df= df[cols[0:4]+ [cols[-1]] +cols[4:12]]
#from the error of not been able to concatenate list means u have to change string to list with []

#to print all elements, we have to use a for loop 
#save to csv
#df.to_csv('modified.csv',index = False)
#save to excel
#df.to_excel('modified.xlsx',index = False)
#df.to_tsv('modified.txt',index = False)
#Filtering of data
#so this is a filter of data using the observation in two columns 
# & 
df.loc[(df['Type 1'] == 'Grass') & (df['Type 2']== 'Poison')]
       
df.loc[(df['Type 1'] == 'Grass') & (df['Type 2']== 'Poison') & (df['HP']>7)]

new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2']== 'Poison') & (df['HP']>7)]
#the old index stays there and i want to reset my index to stop it from showing the 
#old index
new_df = new_df.reset_index(drop = true )

#i want to filter our all the names that contain MEGA
#using the string to search 
[
       #im using the regex expression now 
df.loc[~df['Type 1'].str.contains('Fire|Grass',regex = True)]
# with ~ it was removing what i specified so now lets try it without the sign
df.loc[df['Type 1'].str.contains('Fire|Grass',regex = True)]
       
###now i want to check if type 1 contains MEGA or FIRE
#regex library regular expressions 
#IDENTIFIERS
#\d Any number
#\D anything but a number
#\s space
#\S anything but a space
#\w any character
#\W anything but a character
#. anycharacter except for  anew line
#b the white space around words
#\. a period
#MODIFIERS
# {1,3} we are expecting 1 to 3
# + matcht 1 or more
#? match 0 or more
#* match 0 or more
#$ match the end of a string
#^ matching the beginning of a centnce
## DONT FORGET .+*?[]$ ^(){}| \
#example of range [A-Za-z] it means you are looking for capital A to Z
#example of range [1-5a-qA-Z] this are advanced regular expressions
#example of the regular examples

exampleString = ''' Jessica is 15 years old, and edward is 97, and his
                    grand daughter is 12, Oscar his son is 40'''

agess = re.findall(r'\d{1,3}',exampleString )
names = re.findall(r'[A-Z][a-z]',exampleString )

 df.loc[df['Type 1'].str.contains('fire|grass',flags = re.I,regex = True)]
#what i want to filter must start with pi for example so the character in 
#square bracket means one or more 
 df.loc[df['Name'].str.contains('pi[a-z]*',flags = re.I,regex = True)]
#normally i have an output with the pi in it based on the above code
#but all i have to do is put ^at the begining 
 df.loc[df['Name'].str.contains('^pi[a-z]*',flags = re.I,regex = True)]
## conditional changes for the data frame

df.loc[df[i].str.contains('^\w+$')