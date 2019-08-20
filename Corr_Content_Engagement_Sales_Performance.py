# Import modules
import numpy as np
import pandas as pd
import math
import os
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import neighbors, naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


os.chdir("/Users/liyan.wang/Desktop/Correlation_Content_Engagement_Sales Performance")



# Load files to pandas dataframe
sfdc_share = pd.read_csv("report1544695853372.csv", header=1)
sfdc_opp = pd.read_csv("opportunities.csv", header=0)
sfdc_stage = pd.read_csv("opportunity-history-export.csv", header=0)
mkt_lib = pd.read_csv("library-marketing-content showpad-export-20190709.csv", header=0)
user_act = pd.read_csv("user-activity export.csv", header=0)
top_mkt = pd.read_csv("top-content-marketing.csv", header=0)


def sfdc_data_cleaning(df1, df2, df3):
    sfdc_share_copy = df1[:-7]
    sfdc_share_clean = pd.DataFrame(sfdc_share_copy[['content_name', 'opportunity_id', 'time_spent', 'view_time']]).drop_duplicates(keep='first').sort_values(by=['content_name']).reset_index(drop=True)
    
    sfdc_opp_copy = df2[:-7].drop(['opportunity_type','added_arr_converted_currency','days_from_sqo_to_won',
                                   'owner_name', 'owner_role', 'account_name', 'opportunity_stage'], axis=1)
    
    
    sfdc_stage_copy = df3[:-5].drop(['Stage Change', 'Opportunity Name', 'Owner', 'From Stage', 'Amount Currency', 'Amount', 
                                     'Probability (%)',  'Last Modified',  'Last Modified By'], axis=1)
    
    return sfdc_share_clean, sfdc_opp_copy, sfdc_stage_copy



def engagement_data_cleaning(df1, df2, df3):
    # library marketing content export
    mkt_lib_file = pd.DataFrame(df1.groupby(by='asset name').sum())
    lifetime_engagement = mkt_lib_file[['(page)views', 'likes']]

    # user activity export
    user_act_copy = df2.dropna(subset = ['File Name'])
    recent_engagement = pd.DataFrame(user_act_copy.groupby(by='File Name').agg('sum'))

    # top content marketing export
    top_mkt_file = pd.DataFrame(df3.groupby(by='Display name').sum())
    top_mkt_copy = top_mkt_file[['In-app number of views', 'In-app avg daily view duration (secs)', 
                                 'In-app viewers', 'In-app avg view duration (secs)', 'Recipient number of views',
                                 'Recipient avg daily view duration (secs)', 'Recipient viewers',
                                 'Recipient avg view duration (secs)', 'Shares', 'Social shares']]

    return recent_engagement, lifetime_engagement, top_mkt_copy

def merge_engagement(df1, df2, df3): 
    recent_engagement, lifetime_engagement, top_mkt_copy = engagement_data_cleaning(df1, df2, df3)
    table = pd.concat([recent_engagement, lifetime_engagement, top_mkt_copy], axis=1).reset_index()

    return table


def create_duration(data, var1, var2):
       start_date = pd.to_datetime(data[var1])
       end_date = pd.to_datetime(data[var2])
   
       diff = (end_date - start_date).dt.days
       diff = diff.apply(lambda x : int(x))
   
       return diff


merged_engagement_df = merge_engagement(mkt_lib, user_act, top_mkt).rename(columns={'index': 'content_name','Number of File Views': 'views_recent', 'Number of File Downloads': 'downloads_recent', '(page)views': 'views_lifetime', 'likes': 'likes_lifetime'})
sfdc_clean, opp_clean, stage_clean = sfdc_data_cleaning(sfdc_share, sfdc_opp, sfdc_stage)
opp_combined = pd.merge(opp_clean, stage_clean, how='inner', left_on='opportunity_id', right_on='Opportunity ID').drop('Opportunity ID', axis=1)
sfdc_combined = pd.merge(sfdc_clean, opp_combined, how='inner', on='opportunity_id')


# Insights from invoiced opportunities


invoiced_opp = sfdc_combined.loc[sfdc_combined['To Stage'] == 'Invoiced'].reset_index(drop=True)

a = invoiced_opp[['content_name', 'opportunity_id', 'opportunity_name', 'time_spent', 'opportunity_created_date','opportunity_close_date','added_arr_converted','view_time']].groupby(by=['content_name','opportunity_id', 'opportunity_name']).agg({'time_spent':'sum', 'view_time': 'count','opportunity_created_date':'max','opportunity_close_date':'max','added_arr_converted':'max'})
invoiced_df = pd.DataFrame(a).reset_index().rename(columns={"time_spent": "total_time_spent", "view_time": "#_of_view"})


# Use invoiced id to find out how many stages inviced opporunities went through
invoiced_id = np.array(invoiced_opp[['opportunity_id']].drop_duplicates(keep='first'))
invoiced_id = [invoiced_id[i][0] for i in range(len(invoiced_id))]
invoiced_copy = sfdc_combined.loc[sfdc_combined['opportunity_id'].isin(invoiced_id)].reset_index(drop=True)

#Create count of stages
invoiced_stages = invoiced_copy[['content_name','opportunity_id', 'To Stage']].drop_duplicates(keep='first').groupby(by=['content_name','opportunity_id']).count()
invoiced_stages_col = pd.DataFrame(invoiced_stages).reset_index()
invoiced_df['#_of_stages'] = invoiced_stages_col['To Stage']

merged_invoiced = pd.merge(merged_engagement_df, invoiced_df, how='inner', on='content_name')
merged_invoiced['sales_cycle_length'] = create_duration(merged_invoiced, 'opportunity_created_date', 'opportunity_close_date')

'''
# How many contents used in one opportunity
invoiced_opp_content_used = merged_invoiced[['content_name', 'opportunity_id']].groupby(by='opportunity_id').count()
# How many opportunites used the same content
invoiced_content_opp_used = merged_invoiced[['content_name', 'opportunity_id']].groupby(by='content_name').count()

invoiced_opp_content_used_df = pd.DataFrame(invoiced_opp_content_used).reset_index()
invoiced_content_opp_used_df = pd.DataFrame(invoiced_content_opp_used).reset_index()

sns.lineplot(x='opportunity_id', y='content_name', data=invoiced_opp_content_used_df)

top_content_invoiced = invoiced_opp_content_used_df.loc[invoiced_opp_content_used_df['content_name'] > 8]
print(top_content_invoiced)

sns.lineplot(x='content_name', y='opportunity_id', data=invoiced_content_opp_used_df)

common_content_invoiced = invoiced_content_opp_used_df.loc[invoiced_content_opp_used_df['opportunity_id'] > 4]
print(common_content_invoiced)
'''

# Reorder df columns
merged_invoiced = merged_invoiced[['content_name', 'opportunity_id', 'opportunity_name', 'views_recent', 'downloads_recent', 'views_lifetime',\
       'likes_lifetime', 'In-app number of views', 'In-app avg daily view duration (secs)', 'In-app viewers','In-app avg view duration (secs)', \
       'Recipient number of views', 'Recipient avg daily view duration (secs)', 'Recipient viewers','Recipient avg view duration (secs)', 'Shares', \
       'Social shares','total_time_spent', '#_of_view', 'opportunity_created_date', 'opportunity_close_date','added_arr_converted', '#_of_stages', \
       'sales_cycle_length']]

# Plot correlation matrix
plt.figure(figsize = (10, 8))
sns.heatmap(merged_invoiced.corr(), annot=True, fmt='.2f')

#add opportunity status column
merged_invoiced['status'] = 1



# Insights from lost opportunities


lost_opp = sfdc_combined.loc[sfdc_combined['To Stage'] == 'Lost'].reset_index(drop=True)

b = lost_opp[['content_name', 'opportunity_id', 'opportunity_name', 'time_spent', 'opportunity_created_date','opportunity_close_date','added_arr_converted','view_time']].groupby(by=['content_name','opportunity_id', 'opportunity_name']).agg({'time_spent':'sum', 'view_time': 'count','opportunity_created_date':'max','opportunity_close_date':'max','added_arr_converted':'max'})
lost_df = pd.DataFrame(b).reset_index().rename(columns={"time_spent": "total_time_spent", "view_time": "#_of_view"})

lost_id = np.array(lost_opp[['opportunity_id']].drop_duplicates(keep='first'))
lost_id = [lost_id[i][0] for i in range(len(lost_id))]
lost_copy = sfdc_combined.loc[sfdc_combined['opportunity_id'].isin(lost_id)].reset_index(drop=True)

#count of stages
lost_stages = lost_copy[['content_name','opportunity_id', 'To Stage']].drop_duplicates(keep='first').groupby(by=['content_name','opportunity_id']).count()
lost_stages_col = pd.DataFrame(lost_stages).reset_index()
lost_df['#_of_stages'] = lost_stages_col['To Stage']

merged_lost = pd.merge(merged_engagement_df, lost_df, how='inner', on='content_name')
merged_lost['sales_cycle_length'] = create_duration(merged_lost, 'opportunity_created_date', 'opportunity_close_date')

'''
# How many contents used in one opportunity
lost_opp_content_used = merged_lost[['content_name', 'opportunity_id']].groupby(by='opportunity_id').count()
# How many opportunites used the same content
lost_content_opp_used = merged_lost[['content_name', 'opportunity_id']].groupby(by='content_name').count()

lost_opp_content_used_df = pd.DataFrame(lost_opp_content_used).reset_index()
lost_content_opp_used_df = pd.DataFrame(lost_content_opp_used).reset_index()


sns.lineplot(x='opportunity_id', y='content_name', data=lost_opp_content_used_df)

top_content_lost = lost_opp_content_used_df.loc[lost_opp_content_used_df['content_name'] > 15]
print(top_content_lost)

sns.lineplot(x='content_name', y='opportunity_id', data=lost_content_opp_used_df)

common_content_lost = lost_content_opp_used_df.loc[lost_content_opp_used_df['opportunity_id'] > 20]
print(common_content_lost)
'''

# Reorder df columns
merged_lost = merged_lost[['content_name', 'opportunity_id', 'opportunity_name', 'views_recent', 'downloads_recent', 'views_lifetime',\
       'likes_lifetime', 'In-app number of views', 'In-app avg daily view duration (secs)', 'In-app viewers', 'In-app avg view duration (secs)', \
       'Recipient number of views', 'Recipient avg daily view duration (secs)', 'Recipient viewers', 'Recipient avg view duration (secs)', 'Shares', \
       'Social shares','total_time_spent', '#_of_view', 'opportunity_created_date', 'opportunity_close_date', 'added_arr_converted', '#_of_stages', \
       'sales_cycle_length']]

# Plot correlation matrix
plt.figure(figsize = (10, 8))
sns.heatmap(merged_lost.corr(), annot=True, fmt='.2f')

#add opportunity status column
merged_lost['status'] = 0



# Combine invoiced and lost for creating visualzation of comparison 

df = pd.concat([merged_invoiced, merged_lost], ignore_index=True)

plt.figure(figsize = (10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f')

'''
def overview(df, x_axis, y_axis):
    plt.figure(figsize = (8, 5))
    g = sns.scatterplot(x=x_axis, y=y_axis, hue='status', data=df)
    plt.show()

def cluster_view(df, x_axis, y_axis):
    g = sns.FacetGrid(df, col="status", hue="status")
    g.map(plt.scatter, x_axis, y_axis, alpha=.7)
    g.add_legend()

# Views recent/Views lifetime/Shares vs total time spent
plot1 = overview(df, 'views_recent', 'total_time_spent')
plot2 = cluster_view(df, 'views_recent', 'total_time_spent')

plot3 = overview(df, 'views_lifetime', 'total_time_spent')
plot4 = cluster_view(df, 'views_lifetime', 'total_time_spent')

plot5 = overview(df, 'Shares', 'total_time_spent')
plot6 = cluster_view(df, 'Shares', 'total_time_spent')

# Find out those assets have high total_time_spent but lost the opportunity
insight1 = df.loc[df['total_time_spent'] > 20000]
print(insight1)


# Views recent/Views lifetime/Shares vs # of view
plot7 = overview(df, 'views_recent', '#_of_view')
plot8 = cluster_view(df, 'views_recent', '#_of_view')

plot9 = overview(df, 'views_lifetime', '#_of_view')
plot10 = cluster_view(df, 'views_lifetime', '#_of_view')

plot11 = overview(df, 'Shares', '#_of_view')
plot12 = cluster_view(df, 'Shares', '#_of_view')

# Find out those assets have high # of views but lost the opportunity
insight2 = df.loc[df['#_of_view'] >= 40]
print(insight2)


# Views recent/Views lifetime/Shares vs added arr converted
plot13 = overview(df, 'views_recent', 'added_arr_converted')
plot14 = cluster_view(df, 'views_recent', 'added_arr_converted')

plot15 = overview(df, 'views_lifetime', 'added_arr_converted')
plot16 = cluster_view(df, 'views_lifetime', 'added_arr_converted')

plot17 = overview(df, 'Shares', 'added_arr_converted')
plot18 = cluster_view(df, 'Shares', 'added_arr_converted')

insight3 = df.loc[df['added_arr_converted'] > 800000]
print(insight3)


# Views recent/Views lifetime/Shares vs sales cycle length
plot19 = overview(df, 'views_recent', 'sales_cycle_length')
plot20 = cluster_view(df, 'views_recent', 'sales_cycle_length')

plot21 = overview(df, 'views_lifetime', 'sales_cycle_length')
plot22 = cluster_view(df, 'views_lifetime', 'sales_cycle_length')

plot23 = overview(df, 'Shares', 'sales_cycle_length')
plot24 = cluster_view(df, 'Shares', 'sales_cycle_length')

insight4 = df.loc[df['sales_cycle_length'] < 0]
print(insight4)
'''



# Modeling for opportunity status prediction

x = df.drop(['content_name', 'opportunity_id', 'opportunity_name','opportunity_created_date', 
             'opportunity_close_date','status'],axis=1)
y = df['status']


# Impute missing values
#First check which column(s) exist(s) missing value
#x.isnull().any()

def impute(data, variable):
    # Fill in missing value with most frequent value 
    tmp = np.array([data[variable]]).T

    imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    new_var = imp_freq.fit_transform(tmp)

    new_var = pd.DataFrame(new_var)

    return new_var


impute_list = ['views_recent', 'downloads_recent', 'views_lifetime', 'likes_lifetime', 'In-app number of views','In-app avg daily view duration (secs)', \
       'In-app viewers', 'In-app avg view duration (secs)', 'Recipient number of views', 'Recipient avg daily view duration (secs)', 'Recipient viewers',\
       'Recipient avg view duration (secs)', 'Shares', 'Social shares', 'added_arr_converted']

x_imp = x.copy()

for i in range(len(impute_list)):
    x_imp[impute_list[i]] = impute(x, impute_list[i])


# Create train and test set split at 67/33
x_train, x_test, y_train, y_test = train_test_split(x_imp, y, test_size=0.33, random_state=11)

print('x_train shape is: ', x_train.shape, 'y_train shape is: ', y_train.shape)
print('x_test shape is: ', x_test.shape, 'y_test shape is: ', y_test.shape)


# Normalize indepedent vars
min_max_scaler = preprocessing.MinMaxScaler()
x_train_norm = min_max_scaler.fit_transform(x_train)
x_test_norm = min_max_scaler.fit_transform(x_test)
X_train = np.array(x_train_norm)
X_test = np.array(x_test_norm)
Y_train = np.array(y_train)
Y_test = np.array(y_test)


# Model 1 using default settings using L2 penalty
# Model 2 using L1 penalty
# Model 3 using random forest 'gini' as measurement
# Model 4 using naive bayes
# Model 5 using k-nearest neighbor
# Model 6 using linear discriminant analysis
# Model 7 using neural network default
# Model 8 using neural network logistic activation
# Model 9 using neural network tanh activation
# Model 10 using neural network sgd solver constant learning rate
# Model 11 using neural network sgd solver invscaling learning rate
# Model 12 using neural network sgd solver adaptive learning rate


logReg1 = LogisticRegression(solver='lbfgs')
logReg2 = LogisticRegression(penalty='l1', solver='saga')
rfclf1 = RandomForestClassifier(min_samples_split=3)
nbclf = naive_bayes.GaussianNB()
knnclf = neighbors.KNeighborsClassifier(n_neighbors = 5, weights='distance')
ldclf = LinearDiscriminantAnalysis()
nnclf = MLPClassifier()
nnclf1 = MLPClassifier(activation='logistic')
nnclf2 = MLPClassifier(activation='tanh')
nnclf3 = MLPClassifier(solver='sgd')
nnclf4 = MLPClassifier(solver='sgd', learning_rate='invscaling')
nnclf5 = MLPClassifier(solver='sgd', learning_rate='adaptive')

algorithms = [logReg1, logReg2, rfclf1, nbclf, knnclf, ldclf, nnclf, nnclf1, nnclf2, nnclf3, nnclf4, nnclf5]


def modeling(algo):
    algo = algo.fit(X_train, Y_train)
    predictions = algo.predict(X_test)

    # Use score method to get accuracy of model
    score = algo.score(X_test, Y_test)
    print('Accuracy of {} is {:.4f}'.format(algo, score))

    cm = metrics.confusion_matrix(Y_test, predictions, labels=[1,0])
    print(cm)

    print(classification_report(Y_test, predictions))
    
    return score


acc = []
for i in range (len(algorithms)):
    acc.append(modeling(algorithms[i]))

print(acc)


# Choose randrom forest model as final model
model = rfclf1.fit(X_train, Y_train)
predictions = rfclf1.predict(X_test)
score = rfclf1.score(X_test, Y_test)
cm = metrics.confusion_matrix(Y_test, predictions)

print (rfclf1.feature_importances_)


# Validation
cv_scores = cross_val_score(model, x_imp, y, cv=10)
print("Overall Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))






