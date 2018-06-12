import pandas as pd
dataset = pd.read_csv("vgsales.csv", header=0)
print (dataset.tail(10))

print ("Values in Platform")
print (dataset.Platform.unique())
print ("Values in Genre")
print (dataset.Genre.unique())
print ("Values in Publisher")
print (dataset.Publisher.unique())

platforms = dataset['Platform'].unique()
genres = dataset['Genre'].unique()
platform_val = {}
genre_val = {}
k = 1
for i in platforms:
    platform_val[i] = k
    k = k + 1

k = 1
for i in genres:
    genre_val[i] = k
    k = k + 1
    
dataset["Platform"] = dataset["Platform"].apply(lambda x: platform_val[x])
dataset["Genre"] = dataset["Genre"].apply(lambda x: genre_val[x])
print (dataset["Genre"].head(10))

dataset.drop(['Rank','Name','Publisher'], axis = 1, inplace = True)
print (dataset.head(10))

print ("Values containing NaN")
for column in dataset:
    if dataset[column].isnull().any():
        print (column)
        
dataset['Year'].fillna(dataset['Year'].mean(), inplace = True)
dataset.to_csv('modified_video_game_sales.csv')