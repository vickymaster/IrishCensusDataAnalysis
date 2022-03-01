# IrishCensusDataAnalysis
Data analysis of 2016 Irish Census Data using Python as the data programming language.

The 2016 Irish Census Data has been divided into 15 Themes, each measuring a different population parameter. The selected themes of interest considered for the case study are :

1) Migration, Ethnicity, Religion and Foreign Languages
2) Irish Language
3) Families
4) Education

The project consists of 2 major parts : Exploratory Data Analysis and Statistical Analysis.

Under EDA, I have analyzed different tables in the dataset, performed data visualization and derived some meaningful inferences from these plots. Under Statistical Analysis, I tried to identify Linear Relationship between few parameters using Ordinary Least Squares regression technique. This is followed by performing Hypothesis Testing and finally testing for the assumptions of Linear Regression.

This project helped me to gain great insight from the data anaylsis aspect and I came up with some meaningful conclusions regarding the Irish Census Data.


Following is the code which I developed under this project.


## 1.  INTRODUCTION

The 2016 Irish Census Data is a collection of various features of the population of Ireland collected in 2016 by the Central Statistics Office (CSO). It is the information of each and every person present in Ireland on Census Night. It is conducted after every 5 years (in the years ending with 1 and 6). It helps the government/organizations to plan different community and national services for the betterment/upliftment of the country. The first fully successful census of Ireland was taken in the year of 1821. At that time, the enumerators used to carry notebooks with them and collect information of every individual. Then in 1841, this pattern was changed by introduction of forms and maps for information collection. This was the first modernisation in Census Data, hence it was also called Grear Census of 1841.

The 2016 Census Data has been divided into a total of 15 themes, where each theme measures a different aspect of the papolation parameter. A few examples of themes are Theme 1: Sex, Age and Marital Status; Theme 2: Migration, Ethnicity, Religion and Foreign Languages; Theme 3: Irish Language and so on. All the themes together measure all the aspects of living of each person in Ireland. Apart from this division of data based on themes, the census data is also divided into 3 geographic levels :- county, electoral area and small area. 

For the following case study, the selected themes of interest are as follows:-

* Theme 2: Migration, Ethnicity, Religion and Foreign Languages
* Theme 3: Irish Language 
* Theme 4: Families 
* Theme 10: Education 

The Theme 2 describes the resident population based on their nationality, birthplace, ethnicity, culture, religion and language spoken. Theme 3 indicates the ability of population aged more than 2 years to speak Irish and their frequency of use of Irish language. Theme 4 measures various aspects of families in Ireland such as the family size, number of children in each family, the age of children in each family, number of parents living in the family. Theme 10 describes the intellectual parameter of the society, which is Eeducation / field of study. Under this theme, people who have ceased their education, who haven't ceased their academics, area/field of study by sex and highest level of education completed are studied.

The primary aim of this research study is to analyse the data from above themes, draw some inferences about the population in terms of above parameters and hence conclude with few possible suggestions for the upliftment of the society. Keeping this in mind, efforts have been made to answer questions such as "identify the trends in terms of field of study" , "understand the structure of families in various electoral areas of Ireland" , "does there exists (linear) relationship between the number of people speaking Irish and the country from where they come from?". To achieve this, initially data preprocessing is performed, followed by regression method such as Linear Regression. Expected results include existence of relationship between nationality/birthplace of person and its ability to speak Irish language.




## 2.  DATA CLEANING / PREPROCESSING

# Import all the necessary modules

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# store file path for importing the data

path = "/Users/vivekbulani/Documents/UCD Sem 1 Modules/UCD Python Module/Final Project/"

# import entire datasets for all 3 geographic levels

census_county = pd.read_csv(path+"Census_by_county.csv", thousands=',')
census_electoral = pd.read_csv(path+"census_by_electoral_area.csv", thousands=',')
#census_small_area = pd.read_csv(path+"Census_by_small_area.csv", thousands=',')

# getting the list of all counties included in census data at county geographic level
census_county[['GEOGDESC']]

 

 

#### 2.1 Extracting Table1 of Theme3 i.e Ability to speak Irish - For County level data

Irish_speaking_population = census_county[['GEOGDESC' , 'T3_1YES' , 'T3_1NO' , 'T3_1NS' , 'T3_1T']]

# renaming the column names for better understanding
Irish_speaking_population = Irish_speaking_population.rename(columns={'T3_1YES':'Yes' , 'T3_1NO':'No' , 'T3_1NS':'Not_Stated' , 'T3_1T':'Total' , 'GEOGDESC':'County'})

# setting the index of table to county names
Irish_speaking_population = Irish_speaking_population.set_index('County')

# printing top 5 rows of the table formed

Irish_speaking_population.head()

# checking the datatypes of all columns of the tables created

Irish_speaking_population.dtypes

As all the columns are already of numeric datatype, hence no need of datatype conversion preprocessing.

# checking for presence of null/missing values in the table

Irish_speaking_population.isna().sum()

Hence there are no missing / NA values in the table.

 

#### 2.2 Extracting Table1 of Theme2 i.e Resident population by place of birth and nationality - For County level data

Birthplace_and_Nationality = census_county[['GEOGDESC' , 'T2_1IEBP' , 'T2_1UKBP' , 'T2_1PLBP' , 'T2_1LTBP' , 'T2_1EUBP' , 'T2_1RWBP' , 'T2_1TBP' , 'T2_1IEN' , 'T2_1UKN' , 'T2_1PLN' , 'T2_1LTN' , 'T2_1EUN' , 'T2_1RWN' , 'T2_1NSN' , 'T2_1TN']]

# renaming the column names for better understanding
Birthplace_and_Nationality = Birthplace_and_Nationality.rename(columns={'GEOGDESC':'County' , 'T2_1IEBP':'Ireland_Birthplace' , 'T2_1UKBP':'UK_Birthplace' , 'T2_1PLBP':'Poland_Birthplace' , 'T2_1LTBP':'Lithuania_Birthplace' , 'T2_1EUBP':'Other_EU_Birthplace' , 'T2_1RWBP':'Rest_World_Birthplace' , 'T2_1TBP':'Total_Birthplace' , 'T2_1IEN':'Ireland_Nationality' , 'T2_1UKN':'UK_Nationality' , 'T2_1PLN':'Poland_Nationality' , 'T2_1LTN':'Lithuania_Nationality' , 'T2_1EUN':'Other_EU_Nationality' , 'T2_1RWN':'Rest_World_Nationality' , 'T2_1NSN':'Not_Stated_Nationality' , 'T2_1TN':'Total_Nationality'})

# setting the index of table to county names
Birthplace_and_Nationality = Birthplace_and_Nationality.set_index('County')

# printing top 5 rows of the table formed

Birthplace_and_Nationality.head()

# checking the datatypes of all columns of the tables created

Birthplace_and_Nationality.dtypes

As all the columns are already of numeric datatype, hence no need of datatype conversion preprocessing.

 

 

# checking for presence of null/missing values in the table

Birthplace_and_Nationality.isna().sum() 

Hence there are no missing / NA values in the table.

 

 

 

#### 2.3 Extracting Table3 of Theme10 i.e Field of Study of different Individuals - For Electoral level data 

Education_background = census_electoral[['GEOGDESC', 'T10_3_EDUM', 'T10_3_ARTM', 'T10_3_HUMM', 'T10_3_SOCM', 'T10_3_SCIM', 'T10_3_ENGM', 'T10_3_AGRM', 'T10_3_HEAM', 'T10_3_SERM', 'T10_3_OTHM', 'T10_3_NSM', 'T10_3_TM', 'T10_3_EDUF', 'T10_3_ARTF', 'T10_3_HUMF', 'T10_3_SOCF', 'T10_3_SCIF', 'T10_3_ENGF', 'T10_3_AGRF', 'T10_3_HEAF', 'T10_3_SERF', 'T10_3_OTHF', 'T10_3_NSF', 'T10_3_TF', 'T10_3_NST', 'T10_3_TT']]

# renaming the column names for better understanding
Education_background = Education_background.rename(columns={'GEOGDESC':'Electoral_Area', 'T10_3_EDUM':'Education_teacher_training_Males', 'T10_3_ARTM':'Arts_Males', 'T10_3_HUMM':'Humanities_Males', 'T10_3_SOCM':'Social_sciences_business_law_Males', 'T10_3_SCIM':'Science_mathematics_computing_Males', 'T10_3_ENGM':'Engineering_manufacturing_construction_Males', 'T10_3_AGRM':'Agriculture_veterinary_Males', 'T10_3_HEAM':'Health_welfare_Males', 'T10_3_SERM':'Services_Males', 'T10_3_OTHM':'Other_Males', 'T10_3_NSM':'Not_Stated_Males', 'T10_3_TM':'Total_Males', 'T10_3_EDUF':'Education_teacher_training_Females', 'T10_3_ARTF':'Arts_Females', 'T10_3_HUMF':'Humanities_Females', 'T10_3_SOCF':'Social_sciences_business_law_Females', 'T10_3_SCIF':'Science_mathematics_computing_Females', 'T10_3_ENGF':'Engineering_manufacturing_construction_Females', 'T10_3_AGRF':'Agriculture_veterinary_Females', 'T10_3_HEAF':'Health_welfare_Females', 'T10_3_SERF':'Services_Females', 'T10_3_OTHF':'Other_Females', 'T10_3_NSF':'Not_Stated_Females', 'T10_3_TF':'Total_Females', 'T10_3_NST':"Not_Stated_Total", 'T10_3_TT':'Total'})

# setting the index of table to Electoral Area name
Education_background = Education_background.set_index('Electoral_Area')

# printing top 5 rows of the table formed

Education_background.head()

 

# checking the datatypes of all columns of the tables created

Education_background.dtypes

As all the columns are already of numeric datatype, hence no need of datatype conversion preprocessing.

 

# checking for presence of null/missing values in the table

Education_background.isna().sum()

Hence there are no missing / NA values in the table.

 

 

#### 2.4 Extracting Table1 of Theme4 i.e Families by size of Family - For Electoral level data 

Family_size =  census_electoral[['GEOGDESC', 'T4_1_2PF', 'T4_1_3PF', 'T4_1_4PF', 'T4_1_5PF', 'T4_1_GRE_6PF', 'T4_1_TF']]

# renaming the column names for better understanding
Family_size = Family_size.rename(columns={'GEOGDESC':'Electoral_Area', 'T4_1_2PF':'2_persons', 'T4_1_3PF':'3_persons', 'T4_1_4PF':'4_persons', 'T4_1_5PF':'5_persons', 'T4_1_GRE_6PF':'6_or_more_persons', 'T4_1_TF':'Total_persons'})

# setting the index of table to Electoral Area name
Family_size = Family_size.set_index('Electoral_Area')

# printing top 5 rows of the table formed

Family_size.head()

# checking the datatypes of all columns of the tables created

Family_size.dtypes

As all the columns are already of numeric datatype, hence no need of datatype conversion preprocessing.

 

# checking for presence of null/missing values in the table

Family_size.isna().sum()

Hence there are no missing / NA values in the table.

 

 

## 3.  EXPLORATORY DATA ANALYSIS

#### 3.1 Analysing Irish_speaking_population table 

# printing the summary statistics of the tables (such as minimum, maximum, mean, median, quantiles, etc) 

Irish_speaking_population.describe()

Irish_speaking_population.boxplot(column=["Yes","No"], fontsize=13)

Irish_speaking_population.boxplot(column="Not_Stated", fontsize=13)

Irish_speaking_population.boxplot(column="Total", fontsize=13)

Irish_speaking_population[Irish_speaking_population['Yes']>150000]

(Irish_speaking_population['Yes']/Irish_speaking_population['Total']).sort_values(ascending=False)

* From above outputs of describe( ) and boxplots, we can see that minimum number of people who speak Irish in any county is 12300 and maximum is 179317 (which is for County Cork). Second highest area with largest number of Irish speaking population is Dublin City.

* If we check the proportion of people who speak Irish in each county, then we see Galway has the highest proportion with approximately 50% people staying there speaking Irish. Dublin City has the lowest with only approx 30% people speaking Irish. This might be because most of the foreigners who come to Ireland stay in Dublin City. Hence despite being largely populated, it has not highest number of Irish speaking individuals. Rather there maybe high mixture of people from different countries.

* 25% counties have less than 29473 Irish speaking population. and 75% counties have less than 74538 Irish speaking people.

* As compared to number of individuals either speaking Irish or not, there are considerable amount of people whose status for Irish speaking is not stated. This might be because maybe people don't want to tell whether they know Irish or not.

 

 

Irish_speaking_population['Yes'].hist(bins=30)
plt.xlabel("Total number of people speaking Irish",size=13)
plt.ylabel("Frequency",size=13)
plt.title("Histogram of Irish Speaking Population in different County regions")

From above histogram, we can see that Total Irish speaking population has a Right Skewed Distribution with 2 extreme outliers on right. 

 

 

#### 3.2 Analysing Birthplace_and_Nationality table  

# printing the summary statistics of the tables (such as minimum, maximum, mean, median, quantiles, etc) 

Birthplace_and_Nationality.describe()

Birthplace_and_Nationality.boxplot(column=['Ireland_Birthplace', 'Ireland_Nationality'])

Birthplace_and_Nationality.boxplot(column=['UK_Birthplace', 'Poland_Birthplace', 'UK_Nationality', 'Poland_Nationality'], rot=45)

Birthplace_and_Nationality.boxplot(column=['Other_EU_Birthplace', 'Other_EU_Nationality'])

(Birthplace_and_Nationality['Ireland_Birthplace']/Birthplace_and_Nationality['Total_Birthplace']).sort_values(ascending=False)

(Birthplace_and_Nationality['Rest_World_Birthplace']/Birthplace_and_Nationality['Total_Birthplace']).sort_values(ascending=False)

Birthplace_and_Nationality[Birthplace_and_Nationality['Ireland_Birthplace']>250000].sort_values(ascending=False, by='Ireland_Birthplace')

Birthplace_and_Nationality[Birthplace_and_Nationality['UK_Birthplace']>20000].sort_values(ascending=False, by='UK_Birthplace')

* From above outputs of describe( ) and boxplots, we can see that minimum number of people(in any county) who are born in Ireland is 25901 and maximum is 419158 (which is for Dublin City). Second highest area with largest number of Irish born population is Cork County.

* Also if we see then Dublin City ranks second in number of people who are born in UK followed by Cork County.

* In terms of Other_EU population (i.e individuals who are from outside EU), Dublin City again has highest number of such people.

* Hence from all this, we can say that Dublin City has mixture of both Irish as well as Outside country population.

* Now if we compare the proportion of people who are born in Ireland, we see that County Offaly has highest number of Irish people (approx 88%) and Galway City has lowest (approx 74%). Dublin City also had very low proportion of Irish individuals (approx 78%).

* If we compare the proportion of people in different counties who have came from Rest of the World, Dublin City has highest (approx 10%) (The same result/conclusion was made above while analysing Irish_speaking_population table). Galway City ranks two with approx 0.9%.

* From the output of boxplots derived above, we can infer that Ireland_Birthplace is slightly deviated from normal distribution, it is slightly Right Skewed data. But Ireland_Nationality is almost normally distributed.

* In general there are not much outliers for any column(maximum 5). So most of the data in the table follows the general trend.

* There are considerable amount of people for whom Nationality is not stated/recorded. Hence efforts and care must be taken to keep the record of nationality of every individual.


 

#### 3.3 Analysing Education_background table  

(Education_background['Not_Stated_Total']/Education_background['Total']).sort_values(ascending=False)

((Education_background['Not_Stated_Total']/Education_background['Total'])>0.5).sum()

From above output, we see that there is considerably high amount of people for whom the area of study is not stated/recorded.

For example Cork City North West area has highest proportion of "Not_Stated" value (approx 71%). In total, 87 areas have "Not_Stated" proportion of population greater than 50%. This depicts lack of information in this particular aspect.

 

labels = ['Education_teacher_training', 'Arts', 'Humanities', 'Social_sciences_business_law', 'Science_mathematics_computing', 'Engineering_manufacturing_construction', 'Agriculture_veterinary', 'Health_welfare', 'Services', 'Other']
males = Education_background.iloc[:, 0:10].loc['Dundrum']
females = Education_background.iloc[:, 12:22].loc['Dundrum']

x = np.arange(1,len(labels)+1)

width = 0.175

# plot data in grouped manner of bar type
plt.bar(x-width, males, width, color='cyan')
plt.bar(x+width, females, width, color='green')
plt.xticks(x, labels, size=10, rotation=90)
plt.xlabel("Education Category", size=13)
plt.ylabel("Number of People", size=13)
plt.legend(["Males", "Females"])
plt.title("Grouped Bar Plot Depicting Field of Study by Sex for Dundrum Area", size=13)
plt.show()


 

From above grouped bar chart, we can clearly see that in Dundrum electoral area, highest number of individuals are from "Social Sciences, Business and Law" field. For females, the second highest sector is "Health and Welfare" closely followed by "Education, Teacher and Training" sector. Whereas for males, the second most popular sector is "Engineering, manufacturing and construction" sector closely followed by "Science, mathematics and computing". 

In general we can see that in most of the fields, female population is more compared to male.

 


 

#### 3.4 Analysing Family_size table

 

Family_size.boxplot(column=['2_persons','3_persons','4_persons','5_persons','6_or_more_persons'],figsize=(10,5))

From above boxplot we can infer that in most of the electoral areas, large number of people live in pairs (family size = 2). Proportion of population living with family size 3 and 4 are almost same. There are very few people who live in large  families (comprising of 6 persons or more).

x=[3,7,10,12,14]
y=[4,5,8,11,13]
s=Family_size.loc['Dundrum'].iloc[0:5]

s.values

plt.scatter(x,y,s=s.values)
plt.xlim(0,18)
plt.ylim(0,15)

From above plot, we can see the size of the circle is proportional to number of families of each family size type. Hence for family size = 2, we have largest cricle indicating that there are highest number of families having family size = 2. The circles for family size 3 and 4 are almost same. The smallest is for 6/more persons in family.

 
 
 

## 4.  STATISTICAL ANALYSIS

### 4.1 Identifying if there is any Linear Relationship between ability to speak Irish Langugage and Nationality / Birthplace 

# concatenate the two tables created above i.e Irish_speaking_population and Birthplace_and_Nationality
# concatenation is done using inner join (on county index)

Irish_and_Nationality_merged = pd.concat([Irish_speaking_population , Birthplace_and_Nationality], axis=1, join="inner")

 

# print top 5 rows of the merged data

Irish_and_Nationality_merged.head()

 

# find the correlation of all columns with "Yes" colummn (i.e with number of people who speak Irish).
# then sort the values in decreasing order to identify the columns which are highly related to our column of interest.

Irish_and_Nationality_merged.corr()['Yes'].abs().sort_values(ascending = False)

 


 

##### Building a Linear Model using OLS

# pick top few columns/predictors having high correlation with response variable ("Yes")
data = Irish_and_Nationality_merged[['Ireland_Nationality' , 'Ireland_Birthplace' , 'Poland_Birthplace' , 'Poland_Nationality' , 'UK_Nationality' , 'UK_Birthplace' , 'Rest_World_Birthplace', 'Other_EU_Birthplace' , 'Other_EU_Nationality']]

# necessary to prepend column of 1's to X in order to consider the intercept term
data = sm.add_constant(data)

# defining the response variable
target = Irish_and_Nationality_merged['Yes']

# performing the splitting of entire data into training and testing data in the ratio of 80:20
# to prevent overfitting and to effectively evaluate the model performance.
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0) 

# fit / train the model on training data

mod = sm.OLS(y_train, X_train, axis=1).fit()

 

# print all the estimated coefficient values (i.e intercept term and slope parameter for all variables)

mod.params

Hence our equation is of the form :-

Irish_Yes = β0 + β1 * (Ireland_nationality) + β2 * (Ireland_Birthplace) + β3 * (Poland_Birthplace) + β4 * (Poland_Nationality) + β5 * (UK_Nationality) + β6 * (UK_Birthplace) + β7 * (Rest_World_Birthplace) + β8 * (Other_EU_Birthplace) + β9 * (Other_EU_Nationality)

As per above estimated values for coefficients, we can say that if Ireland_nationality increases by 100 units(people) then number of Irish speaking individuals increases by approximately 72 people. Similarly, for increase in people born in Poland by 100 units, we expect increase in Irish spreaking individuals by approximately 3471.

 

 

# use the fitted model to predict for testing data

y_pred = mod.predict(X_test) 

# get the MSE, RMSE values for testing data

print("Mean Square Error =",metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error =",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

 


### 4.2 Performing Hypothesis Testing

# obtain the 95% confidence intervals for all the estimated coefficients

mod.conf_int()

The confidence intervals can be interpreted as follows :-
If the population from which this sample was drawn was sampled 100 times then approximately 95 of those confidence intervals would contain the "true" coefficient (i.e coefficient lies in that interval)

 

 

# obatin the p-values for all estimated coefficients.
# these values simply represent the probability that the coefficient is actually zero 

mod.pvalues

These p-values can now be used for hypothesis testing for each coefficient where

H0 (null hypothesis) : the predictor variable is insignificant (i.e βi = 0) and

Ha (alternative hypothesis) : the predictor variable is significant (i.e βi != 0)


 

 

# obtain the R-squared value of the model.

mod.rsquared 

It tells the proportion of variance explained by the model. It lies in the interval [0,1].
Closer the R-squared value to 1, better is the model because it means that more variance is explained by the model

 

 

# print the entire summary of the linear model

mod.summary()

From all the above results, we can infer following information :-

* By looking at the estimated values of coefficients (β's), we can clearly say that interpreting these estimated values is pointless. For example, there is a negative value for coefficient of Ireland_Birthplace. But logically this seems to be incorrect as it is highly impossible that people who are born in Ireland don't know to speak Irish.

* Secondly the mean squared error (MSE) and root mean squared error (RMSE) values for testing data are very high.

* Thirdly the p-values for all variables are high (greater than 0.05). Hence this says that none of the predictor variables are significant in determining the number of Irish speaking population. This seems to be logically incorrect.

* Also the value for R-squared and adjusted R-squared are very high (near to 1). This says that almost all of the variance is explained by the model.

* After analysing all these statistical information,  we can finally infer that the model mentioned above explains a lot of variation within the data but is not significant (model is worthless).


 

### 4.3 Testing assumptions of Linear Regression

 

##### Linearity Assumption 

* It states that there must be a linear relationship between the response variable and each predictor variable.

* This assumption can be checked by plotting the component residual plots and added variable plots.

print(sm.graphics.plot_ccpr(mod, "Ireland_Nationality"))

print(sm.graphics.plot_ccpr(mod, "Poland_Birthplace"))

print(sm.graphics.plot_ccpr(mod, "Ireland_Birthplace"))

From all the above plots (and similarly plotting for rest of the variables), we can clearly see that there exists strong linear relationship between the response variable ("YES") and each predictor variable. Hence we can say that this assumpotion is not violated.

 

##### Correlation in Errors (also called Multicollinearity) 

* It states that there should not be any correlation between the predictor variables. Otherwise this may affect the linear model as correlated variables add up their effect. This may lead to masking the effect of other variables.

* This assumption can be checked by performing Durbin - Watson Test. The Durbin - Watson test statistic should ideally be 2 (it indicates that there is no autocorrelation). Value less than 2 indicates positive autocorrelation and value greater than 2 means there is negative autocorrelation. The value for test statistic lies in the range [0,4].

* For practocal purposes, if the Durbin - Watson test statistic is between 1.5 and 2.5, then it not a matter of serious concern.

res = mod.resid
sm.stats.stattools.durbin_watson(res, axis=0)

We can see that Durbin - Watson test statistic is 2.1, which lies in the acceptable range of 1.5 - 2.5. Hence we can say that there is no serious collinearity between predictor variables.

 

 

### 4.4 Anova Table 

from statsmodels.stats import anova
import statsmodels.formula.api as smf
mod1 = smf.ols("Yes ~ Ireland_Nationality + Ireland_Birthplace + Poland_Birthplace + Poland_Nationality + UK_Nationality + UK_Birthplace + Rest_World_Birthplace + Other_EU_Birthplace + Other_EU_Nationality", Irish_and_Nationality_merged).fit()
# print type II ANOVA
print(anova.anova_lm(mod1, typ=2))

We can again see that the p-values for all the predictors are greater than 0.05, hence indicating that none of the predictor variables is significant in linearly predicting the Irish speaking population.

 

 

### 4.5 Trying Linear Regression with new parameters

# finding the correlation of all the columns in county_census dataset with our response variable "T3_1YES" 
# which describes the number of people who speak Irish
# then sort the correlation values in descending order and take only top 20 values.

census_county.corr()['T3_1YES'].abs().sort_values(ascending = False).head(20)

Now, after looking at the description of all these columns, we finally consider following columns for linear regresion:- 

* T10_3_EDUT (Education and teacher training - Total)

* T10_3_EDUF (Education and teacher training - Females)

* T10_3_ENGT (Engineering, manufacturing and construction - Total)

* T10_3_HEAF (Health and welfare - Females)

* T10_1_20T(Age 20 - Total)

* T10_1_20F(Age 20 - Females)

* T10_3_ENGM(Engineering, manufacturing and construction - Males)

# extract the above mentioned columns and create a new dataframe of predictor varibales
subset_data = census_county[['T10_3_EDUT' , 'T10_3_EDUF' , 'T10_3_ENGT', 'T10_3_HEAF' , 'T10_1_20T' , 'T10_1_20F', 'T10_3_ENGM']]

# necessary to prepend column of 1's to X in order to consider the intercept term
subset_data = sm.add_constant(subset_data)

# defining the target variable
new_target = census_county['T3_1YES']

# performing the splitting of entire data into training and testing data in the ratio of 80:20
X_train, X_test, y_train, y_test = train_test_split(subset_data, new_target, test_size=0.2, random_state=0)

# fitting a linear model to the new data

mod1 = sm.OLS(y_train, X_train, axis=1).fit()

# printing out the summary of newly trained linear model

mod1.summary()

 

 

Again we see that for all the predictor variables, p-values are higher than 0.05, thus indicating that none of these predictors are significant. But the R-squared and adjusted R-squared values are near to 1.

Hence we can say that this model is also of no use.

Thus we conclude that there maybe the case that there is more complex relation between the number of people speaking Irish language and predictor variables.

 

 

## 5.  CONCLUSION

From all the above data analysis on 2016 Irish Census Data, we can infer following information :-

1) There exists a strong relationship between the predictor variables (such as Nationality/Birthplace) and number of people who speak Irish. But this relationship cannot be properly explained by a linear model. A more complex model needs to be used for understanding the relationship between them. This model can then be used to predict the number of people who speak Irish in a county given the values of predictor variables.

2) This can also help the government to plan and implement the policies such that more number of individuals living in Ireland can speak Irish. Irish being the mother tongue and national language, it would be better if more and more number of individuals can understand and speak Irish.

3) As seen above, Dublin City has very less proportions of Irish speaking population compared to any other county. Thus measures should be taken such as free / subsised Irish Training classes, providing job opportunities to the people who know Irish, giving incentives to such people and so on. By all such measures, we can see an increase in Irish speaking individuals over the coming years.

4) As seen above, Dublin City and Galway City have the lowest proportion of population who are born in Ireland and high number of population from rest of the world. This shows that more and more number of people are migrating to these cities. But if this continues to happen, then would occur an imbalance in population in cities; that is more number of individuals would be jut living in popular cities leading to highly crowded city areas and comparatively low population in other areas of Ireland. To overcome this, government should take measures to promote migration to other parts of Ireland as well, such as giving incentives to people who shift/migrate to less populated counties, gradually increasing the facilities like transportation, food, jobs, etc. in such areas. So that people are attracted towards such areas also and they do think to live there rather than just staying in city areas.

5) As seen in point 3), there are considerable amount of people for whom Nationality is not stated/recorded. Hence efforts should be taken to overcome this and care must be taken to keep the record of nationality of every individual. Having such information will definitely help the government or other organizations to make plans/policies by considering population percentage various different countires.

6) We see that when we compare the number of males and females based on their field of study, the general trend is that in most fields, females are more compared to males. This may help the government/organizations to create appropriate job opportunities so that females can easily work in the field they have studied.

7) Also when analysing the Education_Background data, we can see that for almost 87 electoral areas, there are no records of field of study of people living there. This may heavily impact as such huge loss of critical information leads to lack of capability of taking further steps/actions. From the education background of the mass, government can find what kind of fields of study are more common and which ones are upcoming/trending. This further helps in developing the job market accordingly, thus increasing the financial status of the society. Also if people get job in the same field in which they have studied, then they would definitely bring back great benefits and revolutions.

8) We can study education aspect along with the employment sector and find relation between the two.

9) We can check that most common type of families in different electoral areas is that having size of 2. Hence it can be said that most people prefer to live in pair/couple and not in big families. That is, nuclear families are more preferred by most of the population.
