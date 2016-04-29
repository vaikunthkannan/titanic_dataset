import numpy as np
import pandas as pd
import csv as csv
import seaborn as sns


import matplotlib.pyplot as plt

raw_data = csv.reader(open("C:/titanic/titanic_data.csv", "rb"))

header = raw_data.next()

titanic_data = []
for row in raw_data:
    titanic_data.append(row)
titanic_data = np.array(titanic_data)

#Proportion of males and females that survived
women_only = titanic_data[0::,4] == "female"

men_only = titanic_data[0::,4] != "female"


women_onboard = titanic_data[women_only,1].astype(np.float)
men_onboard = titanic_data[men_only,1].astype(np.float)

women_survived = np.sum(women_onboard)/np.size(women_onboard)
men_survived = np.sum(men_onboard)/np.size(men_onboard)



#Splitting the fare into three classes
fare_ceiling = 40
titanic_data[titanic_data[0::,9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0

fare_bracket = 10
num_of_price_brackets = fare_ceiling/fare_bracket

num_of_classes = len(np.unique(titanic_data[0::, 2]))
survived_table = np.zeros((2,num_of_classes,num_of_price_brackets))

for i in xrange(num_of_classes):
    for j in xrange(num_of_price_brackets):
        
        survived_women = titanic_data[ (titanic_data[0::,4] == "female") & (titanic_data[0::,2].astype(np.float) == i+1) \
        & (titanic_data[0:,9].astype(np.float) >= j*fare_bracket) & (titanic_data[0::,9].astype(np.float) <(j+1)*fare_bracket), 1]
        
        survived_men = titanic_data[ (titanic_data[0::,4] != "female") & (titanic_data[0::,2].astype(np.float) == i+1) \
        & (titanic_data[0:,9].astype(np.float) >= j*fare_bracket) & (titanic_data[0::,9].astype(np.float) <(j+1)*fare_bracket), 1]

        survived_table[0,i,j] = np.mean(survived_women.astype(np.float))
        survived_table[1,i,j] = np.mean(survived_men.astype(np.float))

survived_table[survived_table != survived_table] = 0
print survived_table




#Using pandas

titanic_df = pd.read_csv("C:/titanic/titanic_data.csv", header = 0)

def plot_survival(data,variable):
    data = data.groupby(variable).mean().reset_index()
    data.plot(kind='bar',x=variable,y='Survived', legend = True)
    plt.title("Survival rate for {0}".format(variable))
    plt.xlabel(variable)
    plt.ylabel('Survived')
    plt.show
    
print plot_survival(titanic_df, "Pclass")
print plot_survival(titanic_df, "Sex")

#Lets find all missing age values

#print titanic_df[titanic_df["Age"].isnull()][["Sex","Pclass","Age"]]
titanic_df["Gender"] = titanic_df["Sex"].map( {"female": 0, "male": 1} ).astype(int)
titanic_df["AgeFill"] = titanic_df["Age"]
#Assigning the median age to missing age values


median_ages = titanic_df.groupby(["Gender","Pclass"]).median()["Age"]

for i in range(0,2):
    for j in range (0,3):
        titanic_df.loc[(titanic_df.Age.isnull()) & (titanic_df.Gender == i) & (titanic_df.Pclass== j+1),\
        "AgeFill"] = median_ages[i, j+1]
        
        
print titanic_df[ titanic_df["Age"].isnull() ][["Gender","Pclass","Age","AgeFill"]].head(10)

print sns.barplot(x="Sex", y="AgeFill", data = titanic_df)

# Relation between point of embarcation and survival

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
print sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
print sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)

embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
print sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
# Comparison of Age and Pclass and Gender
print sns.barplot(x="Sex", y="Pclass", data = titanic_df)

#Did people travelling alone have higher chances of survival

titanic_df["Family"] =  titanic_df["Parch"] + titanic_df["SibSp"]

titanic_df["Family"].loc[titanic_df["Family"] > 0] = 1
titanic_df["Family"].loc[titanic_df["Family"] == 0] = 0


titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)

fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))
print sns.countplot(x="Family", data=titanic_df, order=[1,0], ax=axis1)


family_av = titanic_df[["Family", "Survived"]].groupby(["Family"],as_index=False).mean()
print sns.barplot(x="Family", y="Survived", data=family_av, order=[1,0], ax=axis2)

axis1.set_xticklabels(["With Family","Alone"], rotation=0)

#Comparing Old Age values and New Age values
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Old Age values - Titanic')
axis2.set_title('New Age values - Titanic')
print titanic_df['Age'].hist(ax=axis1)

print titanic_df["AgeFill"].hist(ax=axis2)

