#!/usr/bin/env python
# coding: utf-8

# # CREDIT EDA CASE STUDY

# In[293]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# # Dataset Loading, Data cleaning and Analysis

# In[294]:


appl_df1 = pd.read_csv("application_data.csv")
prev_appl_df2 = pd.read_csv("previous_application.csv")


# Understanding the Data

# In[296]:


appl_df1.head()


# In[297]:


print(appl_df1['NAME_CONTRACT_TYPE'].unique())
print(appl_df1['TARGET'].unique())
print(appl_df1['CODE_GENDER'].unique())
print(appl_df1['NAME_TYPE_SUITE'].unique())


# In[298]:


appl_df1.shape


# In[299]:


appl_df1.isnull().sum()


# In[300]:


missing = appl_df1.isna().any()
na_col1 = missing[missing].index
na_col1


# In[301]:


appl_df1.duplicated().sum()


# In[302]:


appl_df1.describe()


# In[463]:


def misplaced_data_percent(df3):
    data = pd.DataFrame(columns=['Division','Percentage'])
    for i in df3.columns:
        if df3[i].isna().values.any():
            percentage = ((100*df3[i].isna().sum()/len(df3)))
            data = data.append({'Division' : i, 'Percentage' : percentage}, ignore_index=True)
    return data


# In[464]:


data = misplaced_data_percent(appl_df1)
data.sort_values('Percentage', ascending = False)


# In[310]:


prev_appl_df2.head()


# In[311]:


prev_appl_df2.shape


# In[312]:


prev_appl_df2.info()


# In[313]:


prev_appl_df2.isna().sum()


# In[465]:


missingvalues1 = prev_appl_df2.isna().any()
na_columns = missingvalues1[missingvalues1].index
print(na_columns)


# In[317]:


prev_appl_df2.duplicated().sum()


# In[320]:


def misplaced_data_percent(df3):
    data = pd.DataFrame(columns=['Division','Percentage'])
    for i in df3.columns:
        if df3[i].isna().values.any():
            percentage = (df3[i].isna().sum()*100)/(df3.shape[0])
            data = data.append({'Division' : i, 'Percentage' : percentage}, ignore_index=True)
    return data


# In[321]:


data = misplaced_data_percent(prev_appl_df2)
data.sort_values('Percentage', ascending = False)


# In[322]:


prev_appl_df2.drop(['RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED'],inplace = True, axis = 1)
prev_appl_df2.dropna(subset = ['PRODUCT_COMBINATION','AMT_CREDIT'],inplace = True, axis =0)


# In[323]:


data = misplaced_data_percent(prev_appl_df2)
data.sort_values('Percentage', ascending = False)


# In[324]:


numeric_type_cat = prev_appl_df2 .select_dtypes(include= ['int64','float64'])
numeric_type_cat


# In[338]:


plt.figure(figsize= (16,12))
sns.heatmap(numeric_type_cat.corr(), cmap = 'coolwarm', annot = True)
plt.show()


# In[327]:


print(prev_appl_df2['NAME_TYPE_SUITE'].value_counts())
prev_appl_df2['NAME_TYPE_SUITE'].value_counts().plot(kind = 'bar')


# In[328]:


prev_appl_df2['NAME_TYPE_SUITE'].unique()


# In[329]:


prev_appl_df2['NAME_TYPE_SUITE'].isnull().sum()


# In[330]:


prev_appl_df2['NAME_TYPE_SUITE'].fillna(prev_appl_df2.NAME_TYPE_SUITE.mode(),inplace = True)
prev_appl_df2['AMT_DOWN_PAYMENT'].fillna(prev_appl_df2.AMT_DOWN_PAYMENT.mean(), inplace = True)
prev_appl_df2['RATE_DOWN_PAYMENT'].fillna(prev_appl_df2.AMT_DOWN_PAYMENT.mean(), inplace = True)


# In[334]:


print(prev_appl_df2['NAME_TYPE_SUITE'].value_counts())


# In[336]:


previous_merge_data= pd.merge(appl_df1[['SK_ID_CURR','TARGET']], prev_appl_df2,  on=['SK_ID_CURR'], how = 'left')
previous_merge_data.head(5)


# In[337]:


previous_merge_data.info()


# In[480]:


sns.countplot(previous_merge_data['NAME_CONTRACT_STATUS'],hue = previous_merge_data['TARGET'], saturation = 0.9,palette = 'afmhot')
plt.title("Previous Loan application and payment details")
plt.show()


# In[374]:


plt.figure(figsize =(12,2))
sns.boxplot(previous_merge_data['AMT_DOWN_PAYMENT'],  saturation = 0.9,palette = 'afmhot')
plt.show()


# In[373]:


plt.figure(figsize =(12,2))
sns.boxplot(previous_merge_data['AMT_ANNUITY'],  saturation = 0.9,palette = 'afmhot')
plt.show()


# In[387]:


previous_merge_data = previous_merge_data[previous_merge_data['AMT_ANNUITY']<np.nanpercentile(previous_merge_data['AMT_ANNUITY'],99)]
previous_merge_data = previous_merge_data[previous_merge_data['AMT_DOWN_PAYMENT']<np.nanpercentile(previous_merge_data['AMT_DOWN_PAYMENT'],99)]


# In[388]:


plt.figure(figsize =(12,2))
sns.boxplot(previous_merge_data['AMT_DOWN_PAYMENT'], saturation = 0.9,palette = 'afmhot')
plt.show()


# In[389]:


plt.figure(figsize =(12,2))
sns.boxplot(previous_merge_data['AMT_ANNUITY'],  saturation = 0.9,palette = 'afmhot')
plt.show()


# In[390]:


approved_loans = previous_merge_data[previous_merge_data['NAME_CONTRACT_STATUS']=='Approved'].shape[0]
defaulted_in_approved = previous_merge_data[(previous_merge_data['TARGET']==1) & (previous_merge_data['NAME_CONTRACT_STATUS']=='Approved')].shape[0]
print('percentage of previously approved loans which defaulted in current loan scenario:',round((defaulted_in_approved*100/approved_loans),4))


# In[391]:


refused_loans = previous_merge_data[previous_merge_data['NAME_CONTRACT_STATUS']=='Refused'].shape[0]
Not_default_in_refused = previous_merge_data[(previous_merge_data['TARGET']==0) & (previous_merge_data['NAME_CONTRACT_STATUS']=='Refused')].shape[0]
print('percentage of previously refused loans which were non defaulted in current loan scenario:',round((Not_default_in_refused*100/refused_loans),4))


# In[392]:


previous_merge_data[previous_merge_data['TARGET']== 0]['RATE_DOWN_PAYMENT'].quantile([0.5,0.7,0.9,0.95,0.99])


# In[393]:


previous_merge_data[previous_merge_data['TARGET']== 1]['RATE_DOWN_PAYMENT'].quantile([0.5,0.7,0.9,0.95,0.99])


# In[396]:


non_defaulters = appl_df1[appl_df1['TARGET'] ==0]
defaulters = appl_df1[appl_df1['TARGET']==1]
print('NO of non defaulters:',non_defaulters.shape[0])
print('NO of defaulters:',defaulters.shape[0])

print('percent of defaulters:', defaulters.shape[0]*100/(defaulters.shape[0]+non_defaulters.shape[0]))


# In[473]:


sns.countplot(previous_merge_data['TARGET'], saturation = 0.9,palette = 'afmhot')
plt.title('Payment_related')
plt.show()


# In[481]:


sns.countplot(previous_merge_data[previous_merge_data['NAME_CONTRACT_STATUS']=='Refused']['CODE_REJECT_REASON'], hue = previous_merge_data.TARGET, saturation = 0.9,palette = 'afmhot')
plt.title('Reasons for resuing application')
plt.show()


# In[400]:


sns.countplot(previous_merge_data.NAME_CONTRACT_TYPE, hue = previous_merge_data['TARGET'], saturation = 0.9,palette = 'afmhot')
plt.show()


# In[484]:


fig =plt.figure(figsize = (18,4))
ax1 = fig.add_subplot(1,2,1, title = 'Type of Client')
ax2 = fig.add_subplot(1,2,2, title = 'Payment methods')
sns.countplot(previous_merge_data['NAME_CLIENT_TYPE'], hue = previous_merge_data.TARGET, ax=ax1,  saturation = 0.9,palette = 'afmhot')
sns.countplot(previous_merge_data['NAME_PORTFOLIO'], hue = previous_merge_data.TARGET, ax=ax2,  saturation = 0.9,palette = 'afmhot')
plt.xticks(rotation=90)

plt.show()


# In[485]:


fig =plt.figure(figsize = (10,4))
ax2 = fig.add_subplot(1,1,1, title = 'Mode of payment')
sns.countplot(previous_merge_data['NAME_PAYMENT_TYPE'], hue = previous_merge_data.TARGET, ax=ax2, saturation = 0.9,palette = 'afmhot')
plt.xticks(rotation=90)
plt.show()


# In[404]:


def value_wise_loan_default_percentage(df,col):
    new_valuedf = pd.DataFrame(columns=['value','percentage of defaulter'])
    
    for value in df[col].unique():
        default_cnt = df[(df[col] == value) & (df.TARGET == 1)].shape[0]
        total_cnt = df[df[col]==value].shape[0]
        new_valuedf = new_valuedf.append({'value': value, 'percentage of defaulter': (default_cnt*100/total_cnt)}, ignore_index = True)
    return new_valuedf.sort_values(by='percentage of defaulter', ascending = False)


# In[407]:


value_wise_loan_default_percentage(previous_merge_data, 'PRODUCT_COMBINATION')


# In[408]:


previous_merge_data['PRODUCT_COMBINATION']


# In[409]:


value_wise_loan_default_percentage(previous_merge_data, 'NAME_GOODS_CATEGORY')


# In[410]:


value_wise_loan_default_percentage(previous_merge_data, 'CHANNEL_TYPE')


# In[411]:


appl_df1.head()


# In[412]:


value_wise_loan_default_percentage(appl_df1,'NAME_HOUSING_TYPE')


# In[414]:


plt.title('payment details of current loans')
sns.countplot(appl_df1['TARGET'], saturation = 0.9,palette = 'afmhot')
plt.show()


# In[477]:


plt.figure()
sns.countplot(appl_df1['NAME_HOUSING_TYPE'], hue = appl_df1['TARGET'], saturation = 0.9,palette = 'afmhot')
plt.xticks(rotation=90)
plt.title('Type of living vs Target_of defaulters and Non defaulters')
plt.show()


# In[478]:


plt.figure()
sns.countplot(appl_df1['NAME_FAMILY_STATUS'], hue = appl_df1['TARGET'], saturation = 0.9,palette = 'afmhot')
plt.xticks(rotation=90)
plt.title('Individual status vs Target_of defaulters and Non defaulters')
plt.show()


# In[417]:


value_wise_loan_default_percentage(appl_df1,'NAME_FAMILY_STATUS')


# In[418]:


appl_df1['AMT_INCOME_TOTAL']


# In[423]:


print(appl_df1.NAME_INCOME_TYPE.value_counts())
appl_df1.NAME_INCOME_TYPE.value_counts().plot(kind = 'bar')


# In[424]:


appl_df1['CODE_GENDER'].isnull().sum()


# In[426]:


sns.countplot(appl_df1['CODE_GENDER'], hue = appl_df1['TARGET'],saturation = 0.9,palette = 'afmhot')
plt.show()


# In[427]:


appl_df1['AGE'] = round(appl_df1.DAYS_BIRTH//(-365.25),6).astype(int)
appl_df1['AGE'].describe()


# In[428]:


appl_df1['AGE_GROUPS'] = pd.cut(appl_df1['AGE'], bins= [18,25,38,60,80], labels = ['Very young','Young','Moderate Age','Senior citizens'])


# In[457]:


appl_df1['Income_category'] = pd.qcut(appl_df1['AMT_INCOME_TOTAL'], q = [0,0.20,0.40,0.60,0.8,1], labels = ['Very Low','Low','Moderate','High','Very High'])


# In[458]:


appl_df1.head(10)


# In[438]:


Target_variable0 = appl_df1.loc[appl_df1['TARGET']==0]
Target_variable1 = appl_df1.loc[appl_df1['TARGET']==1]


# In[451]:


plt.figure(figsize =(16,8))
plt.subplot(121)
sns.countplot(x = 'TARGET', hue = 'AGE_GROUPS', data = Target_variable0)
plt.subplot(122)
sns.countplot(x = 'TARGET', hue = 'AGE_GROUPS', data = Target_variable1)
plt.show()


# In[431]:


appl_df1['OCCUPATION_TYPE'].isnull().value_counts()


# In[432]:


appl_df1['OCCUPATION_TYPE'].fillna('Unknown', inplace =True)


# In[433]:


value_wise_loan_default_percentage(appl_df1,'OCCUPATION_TYPE')


# In[434]:


value_wise_loan_default_percentage(appl_df1,'NAME_EDUCATION_TYPE')


# In[468]:


appl_df1['AMT_ANNUITY'].plot(kind = 'kde')


# In[459]:


pivot_table1= pd.pivot_table(appl_df1, values = 'TARGET', index = ['CODE_GENDER','Income_category'], columns = ['NAME_EDUCATION_TYPE'], aggfunc = np.mean)
pivot_table1

