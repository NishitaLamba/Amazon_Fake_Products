#!/usr/bin/env python
# coding: utf-8

# # Necessary imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk 
from sklearn.datasets import fetch_20newsgroups
from wordcloud import WordCloud
from nltk.corpus import stopwords
import re as re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from eli5.sklearn import PermutationImportance
import eli5
try:
    from sklearn.feature_selection import VarianceThreshold
except:
    pass  # it will catch any exception here
import warnings
warnings.filterwarnings("ignore")


# # Loading data file

# In[2]:


data = pd.read_pickle('/Users/nishitalamba/Downloads/counter_dataset')


# In[3]:


data.head()


# # Data Exploration

# ## Rows and Columns 

# In[4]:


data.shape


# In[5]:


data.columns


# ## Data Types 

# In[6]:


data.dtypes


# ## Missing Values

# In[7]:


data.isna().sum()


# In[8]:


# Calculating percentage of null values in each column
(data.isnull().sum() * 100 / data.index.size).round(2)


# ## Data Visualization

# In[9]:


data['is_verified_purchase'].value_counts().plot(kind='bar',color='skyblue', figsize=(7,5))
plt.xlabel("is verified purchase")
plt.ylabel("Frequency")
plt.show()


# In[10]:


data['is_vine_voice'].value_counts().plot(kind='bar',color='skyblue', figsize=(7,5))
plt.xlabel("Vine Reviewer")
plt.ylabel("Frequency")
plt.show()


# In[11]:


data.groupby(['rating'])['is_verified_purchase'].value_counts().plot(kind='area',color = 'orange', figsize=(7,5))


# In[12]:


data.groupby(['is_verified_purchase'])['rating'].mean().plot(kind='bar', color='skyblue',figsize=(7,5))
plt.ylabel("rating")


# In[13]:


result1 = data[(data["offer_merchant"] != data["offer_fulfiller"])]


# In[14]:


ax = data['rating'].value_counts().plot(kind='bar',figsize=(10,8))
result1['rating'].value_counts().plot( kind="bar", ax=ax, color="C3")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.title ('For products where seller and supplier are different')
plt.show()


# In[15]:


data['category'].value_counts().plot(kind='barh',color='orange',figsize=(10,8))
plt.xlabel("Category")
plt.ylabel("Number of events")
plt.show()


# In[16]:


data['is_prime_exclusive'].value_counts().plot(kind='bar',color='skyblue',figsize=(10,8))
plt.xlabel("prime exclusive")
plt.ylabel("Count")
plt.show()


# In[17]:


data['is_prime'].value_counts().plot(kind='bar', figsize=(10,8))
plt.xlabel("prime")
plt.ylabel("Count")
plt.show()


# In[18]:


data['is_fresh'].value_counts().plot(kind='bar',color = 'skyblue', figsize=(10,8))
plt.xlabel("fresh")
plt.ylabel("Count")
plt.show()


# In[19]:


# products which are both prime and exclusive
data.groupby(['is_prime'])['is_prime_exclusive'].value_counts().plot(kind='bar', figsize=(10,8))
plt.xlabel("prime")
plt.ylabel("Count")
plt.show()


# In[20]:


result1 = data[(data["is_prime"] == data["is_prime_exclusive"])]
result1.count()


# In[21]:


ax = data['rating'].value_counts().plot(kind='bar',figsize=(10,8))
result1['rating'].value_counts().plot( kind="bar", ax=ax, color="C3")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.title ('For products which are both prime and prime exclusive')
plt.show()


# ### Correlation: To how the extent of relation between columns

# In[22]:


total_corr = data.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)


# In[23]:


upper = total_corr.where(np.triu(np.ones(total_corr.shape), k=1).astype(np.bool))


# In[24]:


upper[(upper > 0.9)].stack()


# In[25]:


import plotly.express as px
fig = px.imshow(total_corr)
fig.show()


# ## Computing date 

# In[26]:


from datetime import datetime
data['date_posted'] = pd.to_datetime(data['date_posted'])
# data['date'] = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')


# In[27]:


data['date_posted']


# In[28]:


a = data.sort_values(by='date_posted')
a["date_posted"].iloc[[0,-1]]


# In[29]:


# Grouping ratings by year
grouped_monthly = data.groupby([data["date_posted"].dt.year])
grouped_monthly["rating"].value_counts().plot(kind="bar", figsize=(15,10))
plt.xlabel("Year")
plt.ylabel("ratings")
plt.title("ratings by year")
plt.show()


# In[30]:


# Grouping ratings by month
grouped_monthly = data.groupby([data["date_posted"].dt.month])
grouped_monthly["rating"].value_counts().plot(kind="bar", figsize=(15,10))
plt.xlabel("month")
plt.ylabel("ratings")
plt.title("ratings by year")
plt.show()


# # Data Manipulation

# Handling numerical and categorical data separately for easy manipulation

# In[31]:


cat = data.select_dtypes(exclude=['int', 'float']) # Categorical columns
num = data.select_dtypes(include=['int', 'float']) # Numerical columns


# In[32]:


text = data[['asin','review_text','brand']] # text columns for NLP


# In[33]:


print(cat.columns)
print(num.columns)


# In[34]:


cat.isna().sum() #Missing values in categorical columns


# ## Missing value imputation 

# Filling NA with mean value of column for each product

# In[35]:


cat['is_add_on'] = cat.groupby('asin').transform(lambda x: x.fillna(x.mean()))


# In[36]:


cat["is_prime"] = cat.groupby("asin").transform(lambda y: y.fillna(y.mean()))


# In[37]:


cat["is_prime_pantry"] = cat.groupby("asin").transform(lambda z: z.fillna(z.mean()))


# In[38]:


cat["is_prime_exclusive"] = cat.groupby("asin").transform(lambda e: e.fillna(e.mean()))


# In[39]:


cat["is_fresh"] = cat.groupby("asin").transform(lambda i: i.fillna(i.mean()))


# In[40]:


cat["has_sns"] = cat.groupby("asin").transform(lambda w: w.fillna(w.mean()))


# Filling NA with maximum occuring value of brand for each product because brand will remain the same for each product despite of different reviews

# In[41]:


cat['brand'] = cat.groupby('asin')['brand'].apply(lambda x:x.fillna(x.value_counts()))


# Filling NA for brand column with 'Unknown', in case none of the product rows have a brand specified

# In[42]:


cat['brand'] = cat.replace(np.nan, 'Unknown', regex=True)


# For price, we can't impute with mean since all products have a very different price range so we fill NA with 0 and create binary column to indicate missing rows

# In[43]:


num['list_price'] = pd.concat([num, num['list_price'].isnull().astype(int).add_suffix('_missing')], axis=1)


# In[44]:


num['list_price'].replace(np.nan, 0, inplace=True)


# In[45]:


cat["num_reviews"] = cat.groupby("asin").transform(lambda w: w.fillna(w.mean()))


# In[46]:


cat = cat.drop(['body','review_text','asin'],1)


# ### Checking for null values

# In[47]:


num.isna().sum()


# In[48]:


cat.columns[cat.isnull().mean() > 0.8]


# In[49]:


num.columns[num.isnull().mean() > 0.8]


# In[50]:


num['list_price'] = num['list_price'].fillna(num['price_low'])
# price high and price low are 100% correlated so we just drop them and impute list price with one of them


# In[51]:


# dropping columns with more than 80% nans
cat = cat.drop(['offer_fulfiller','offer_merchant'], axis=1)
num = num.drop(['lowest_price_new_condition','price_low','price_high'], axis=1)


# In[52]:


cat = cat.astype('str') #converting all categorical columns to string in order to process encoding


# In[53]:


# encoding categorical columns
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

r = cat.apply(le.fit_transform)

encoded_cat = pd.DataFrame(r, 
             columns=cat.columns)


# ## Combined dataframe after data cleaning and manipulation 

# In[54]:


df3 = pd.concat((num,encoded_cat),axis=1)


# In[55]:


df3.isna().sum() # all nans have been dealt with and the data is ready for use


# # Sentiment Analysis

# To classify the reviews as positive and negative reviews

# In[56]:


data1 = pd.concat((text,num,encoded_cat),axis=1)


# In[57]:


data1 = data1[data1['review_text'].notnull()]


# In[58]:


for i in range(0,len(data1)-1):
    if type(data1.iloc[i]['review_text']) != str:
        data1.iloc[i]['review_text'] = str(data1.iloc[i]['review_text'])


# In[59]:


analyzer = SentimentIntensityAnalyzer()


# In[60]:


sentiment = data1['review_text'].apply(lambda x: analyzer.polarity_scores(x))
data1 = pd.concat([data1, sentiment.apply(pd.Series)],1)


# In[61]:


def conditions(data1):
    if (data1['compound'] < -0):
        return 'Negatve'
    elif (data1['compound'] > 0):
        return 'Positive'
    else:
        return 'Neutral'


# In[62]:


data1['sentiment'] = data1.apply(conditions, axis=1)
data1['sentiment'].unique()

data1.sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["pink", "skyblue", "grey"])


# Overall sentiment of the reviews is positive - which also suggests the presence of fake reviews

# In[63]:


def sentiment(n):
    return 1 if n >= 4 else 0

# New Column: Rating_sentiment: binary column based on ratings
data1['rating_sentiment'] = data1['rating'].apply(sentiment)


# In[64]:


data1.rating_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%')


# In[65]:


# Average rating for each group of sentiment 
data1.groupby(['sentiment'])['rating'].mean()


# In[66]:


# data2 = pd.concat((data1['compound'],df3),axis=1)


# In[67]:


# data2 = data2[data2['list_price'].notna()]


# # Developing sentiment classifier

# In[68]:


X = data1['review_text']
y = data1['sentiment']


# In[69]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[70]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
ctmTr = cv.fit_transform(X_train)
X_test_dtm = cv.transform(X_test)


# In[71]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(ctmTr, y_train)


# In[72]:


y_pred_class = model.predict(X_test_dtm)
y_pred_class


# In[73]:


accuracy_score(y_test, y_pred_class)


# # Using only reviews (text data) for identifying fake products

# ### Compiling reviews to create a corpus for each product id

# In[74]:


new = data.groupby(['asin'])['review_text'].apply(','.join).reset_index()


# In[75]:


# Also keeping the count of reviews for each product
count_ = data.groupby('asin').num_reviews.count()
type(count_)
count_ = count_.to_frame()
count_


# In[76]:


new = pd.merge(new, count_, on="asin")
new.head()


# ### Created a function key_sentence to find all the review corpuses which contain the word 'Fake'

# In[77]:


def key_sentence(str1, word):
    result = re.findall(r'([^.]*'+word+'[^.]*)', str1)
    return result

new['filter_sentence_fake']=new['review_text'].apply(lambda x : key_sentence(x,'fake'))
print("\nText with the word 'fake':")
new.head()


# ### Created a function key_sentence to find all the review corpuses which contain the word 'Duplicate'

# In[78]:


new['filter_sentence_duplicate']=new['review_text'].apply(lambda x : key_sentence(x,'duplicate'))
print("\nText with the word 'duplicate':")
new.head()


# ### Merged the 2 columns - one with the word fake and the one with word duplicate in its rows

# In[79]:


new['fake_text'] = new['filter_sentence_fake'].astype(str) + new['filter_sentence_duplicate'].astype(str)


# In[80]:


new.count() # all reviews that contain words fake/duplicate


# In[81]:


fake = new[new["fake_text"].str.len() > 4]


# In[82]:


fake.count() # all products that contain words fake/duplicate in reviews


# #### list of all fake products

# In[83]:


list_fake = (fake['asin'].to_list())       


# In[84]:


fake.fake_text.str.count("fake|duplicate").sum() # all reviews that contain words fake/duplicate     


# In[85]:


wordlist = ['fake','duplicate']

# Calculating total number of times the words fake/duplicate have been used for each product
fake['total_occur'] = (fake['fake_text'].str.count(r'\b|\b'.join(wordlist)))

# Calculating the rate of occurrance by dividing total occurrance by total number of reviews for each product
fake['rate_of_occur'] = fake['total_occur']/fake['num_reviews']


# In[86]:


# Sorting by rate of occurance to get the products highly susceptible to be counterfeit.
fake.sort_values(by=['rate_of_occur'], inplace=True, ascending=False)


# In[87]:


# Products with positive rate of occurrance are most likely duplicate
fake.loc[(fake['rate_of_occur'] > 0)].count()


# In[ ]:





# # Brands susceptible to the risk of counterfeit or gray market selling

# In[88]:


data[data['asin'].isin(list_fake)]['brand'].unique().shape # Number of brands highly susceptible


# In[89]:


data[data['asin'].isin(list_fake)]['brand'].unique() # List of brands highly susceptible


# In[90]:


brands = data[data['asin'].isin(list_fake)]['brand']


# # Extent of being susceptible (product wise)

# Calculating total fake reviews for each brand

# In[91]:


brands = brands.value_counts().rename_axis('unique_values').to_frame('counts') 


# In[92]:


top_counts = brands[(brands > 1000).any(1)]  # filtering for brands where count is >1000
top_counts


# In[93]:


fig = plt.figure(figsize=(4,4), dpi=200)
ax = plt.subplot(111)

top_counts.counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=270, fontsize=3)


# ###  Brands with the highest risk of being counterfiet 

# In[94]:


top_counts1 = brands[(brands > 10000).any(1)] # filtering for brands where count is >10000
top_counts1


# In[95]:


fig = plt.figure(figsize=(2,2), dpi=200)
ax = plt.subplot(111)

top_counts1.counts.plot(kind='pie', ax=ax, autopct='%1.1f%%' ,startangle=400, fontsize=5)


# # Extent of being susceptible

# In[96]:


fake.sort_values(by=['rate_of_occur'], inplace=True, ascending=False)


# In[97]:


fake.head(10)    #top 20


# ### Wordcloud for reviews which indicate counterfeiting 

# In[98]:


all_words = ' '.join([text for text in fake['fake_text']])
Stopwords = ["product","shoe","used","even","tried","one","bought",'love']
from wordcloud import WordCloud
wordcloud = WordCloud(width=900, height=600, random_state=21, stopwords = Stopwords,max_font_size=110, colormap="Reds_r").generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# Clearly indicates fake but also has positive ones like ‘real’ and ‘great’ which suggests fake reviews or spamming
# 

# ## Machine Learning for data filtered for Fake product  

# In[99]:


data3 = pd.concat((text,df3),axis=1)


# In[100]:


fake_products = data3[data3['asin'].isin(list_fake)]
fake_products.head()


# In[101]:


fake_products = fake_products.drop('review_text',1)


# In[102]:


#encoding for feeding into the model
t = fake_products.apply(le.fit_transform)

fake_products = pd.DataFrame(t, 
             columns=fake_products.columns)


# In[103]:


fake_products.head()


# In[104]:


fake_products['is_verified_purchase'].value_counts().plot(kind='bar', color='skyblue',figsize=(7,5))


# ## Checking for correlated columns

# In[105]:


corr = fake_products.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)


# In[106]:


upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool)) # considering only upper traingle of correlation


# In[107]:


upper[(upper > 0.9)].stack() # filtering for columns with correlation greater than 90%


# In[108]:


import plotly.express as px   #interactive correlation plot
fig = px.imshow(corr)
fig.show()


# In[109]:


to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
print(to_drop)


# In[110]:


# dropping columns with correlation greater than 90%
fake_products_ = fake_products.drop(columns=[col for col in df3 if col in to_drop])
fake_products_.head()


# In[111]:


fake_products_ = fake_products_.drop(['asin','review_post_id'],1)


# In[112]:


fake_products_.head()


# Treating ratings column as dependent variable

# In[114]:


X = fake_products_.drop(['rating'],1)
y = fake_products_[['rating']]


# In[115]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ## Gaussian Naive Bayes model to understand feature importance applied to the dataset for fake products only

# In[116]:


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
model = gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)


# In[117]:


from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[118]:


perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# ### Interpreting Permutation Importances
# The values towards the top are the most important features, and those towards the bottom matter least.
# Ratings are most affected by helpful_count, brand name and sentiment score

# # Principal Component Analysis 

# In[119]:


# PCA
from sklearn.decomposition import PCA
# Loop Function to identify number of principal components that explain at least 99% of the variance
for comp in range(fake_products_.shape[1]):
    pca = PCA(n_components= comp, random_state=42)
    pca.fit(fake_products_)
    comp_check = pca.explained_variance_ratio_
    final_comp = comp
    if comp_check.sum() > 0.95:
        break
        
Final_PCA_1 = PCA(n_components= final_comp,random_state=42)
Final_PCA_1.fit(fake_products_)
cluster_df=Final_PCA_1.transform(fake_products_)
num_comps = comp_check.shape[0]
print("Using {} components, we can explain {}% of the variability in the original data.".format(final_comp,comp_check.sum()))


# Clearly shows that all the features can be combined into 1 and can explain 99% variability in the data. This is because the features are highly correlated and contribute to very less variance.

# # Dealing with original data to check which factors affect gray selling

# In[120]:


data12 = pd.concat((data1['sentiment'],df3),axis=1)
data12.head()


# In[121]:


data12['sentiment'] = data12['sentiment'].map(dict(Positive=2, Negatve=0 , Neutral  = 1))
data12.head()


# In[122]:


corr = data12.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)


# In[123]:


upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))


# In[124]:


upper[(upper > 0.8)].stack()


# In[125]:


to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
print(to_drop)


# In[126]:


data2 = data12.drop(columns=[col for col in df3 if col in to_drop])


# ## Gaussian Naive Bayes model to understand feature importance applied to the original dataset 

#  Treating sentiment column as dependent variable for Gaussian naive bayes

# In[127]:


X = data2.drop(['sentiment','review_post_id'],1)
y = data2[['sentiment']]


# In[128]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[129]:


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
model = gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)


# In[130]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# ## Permutation Importance for feature evaluation

# In[131]:


from sklearn.inspection import permutation_importance
imps = permutation_importance(model, X_test, y_test)
print(imps.importances_mean)


# In[132]:


perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# ### Interpreting Permutation Importances
# The values towards the top are the most important features, and those towards the bottom matter least

# In[ ]:





# In[ ]:





# In[ ]:




