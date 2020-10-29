#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
def ignore_warn(*args, **kwargs):
    pass

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
import shap #for SHAP values
from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproducibility

pd.options.mode.chained_assignment = None  #hide any pandas warnings


# In[2]:


#Load the Dataset
df = pd.read_csv("games.csv")
df.head()


# In[3]:


df.dtypes

id: The random id assigned to each game
rated: Whether or not the game was a rated game (affects score)
created_at: time the game was created
last_move_at: time of last move
turns: number of combined turns between black and white
victory_status: checkmate, outoftime, resign
winner: who won (white/black)
increment_code: Game time (Minutes To Start + Seconds Added Per Turn)
white_id: name of white opponent
white_rating: chess rating of white opponent
black_id: name of black opponent
black_rating: chess rating of black opponent
moves: total list of moves
openingeco: set of opening classification moves by Encyclopaedia of Chess Openings (ECO) code
opening_name: name of the opening
onening_ply: number of opening movesWe want to determine whether we can predict the outcome of white or black winning based on various data.
Do certain openings have a higher win percentage?
Is there a correlation between opening type and total number of moves?
Does rating and total number of moves follow the same linear plot?
# In[4]:


#Change the column names to clearer terms
df.columns = ['game_id', 'rated_game?', 'start_time', 'end_time', 'num_of_moves',
             'win_type', 'winner', 'increment_code', 'white_id', 'white_rating',
             'black_id', 'black_rating', 'list_of_moves', 'opening_eco', 'opening_name',
             'num_opening_moves']
#Split the increment code into two columns
df[['time', 'added_time']] = df.increment_code.str.split("+",expand=True)


# In[5]:


#Change values of the categorical variables
df['rated_game?'][df['rated_game?'] == True] = '1'
df['rated_game?'][df['rated_game?'] == False] = '0'

df['winner'][df['winner'] == 'white'] = '1'
df['winner'][df['winner'] == 'black'] = '0'

df['win_type'][df['win_type'] == 'draw'] = '0'
df['win_type'][df['win_type'] == 'outoftime'] = '0'
df['win_type'][df['win_type'] == 'resign'] = '1'
df['win_type'][df['win_type'] == 'mate'] = '3'

df.head()


# In[6]:


df.dtypes


# In[7]:


#Correct the type of data
df['game_id'] = df['game_id'].astype('category')
df['game_id'] = df['game_id'].cat.codes
df['game_id'] = df['game_id'].astype('object')

df['black_id'] = df['black_id'].astype('category')
df['black_id'] = df['black_id'].cat.codes
df['black_id'] = df['black_id'].astype('object')

df['white_id'] = df['white_id'].astype('category')
df['white_id'] = df['white_id'].cat.codes
df['white_id'] = df['white_id'].astype('object')

df['time'] = df['time'].astype('int')

df.dtypes

Dropping the start and end time variables as the given data is not fomatted correctly. List of moves has too many categories and doesn't help with our analysis. Opening name is grouped better with opening_eco, game_id, white_id, and black_id are not important as we are not interested in individual user scores or specific games.
# In[8]:


#drop the start and end time variables as the given data is not formatted correctly
#df = df.drop(['start_time','end_time', 'list_of_moves', 'opening_name', 'white_id', 'black_id'], axis = 1)

df.head()


# In[9]:


#Create dummy variables
df = pd.get_dummies(df, drop_first=True)
df.head()


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('winner_1', 1), df['winner_1'], test_size = .2, random_state=10) #split the data


# In[11]:


model = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)


# In[12]:


estimator = model.estimators_[1]
feature_names = [i for i in X_train.columns]

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values


# In[13]:


y_predict = model.predict(X_test)
y_pred_quant = model.predict_proba(X_test)[:, 1]
y_pred_bin = model.predict(X_test)


# In[14]:


confusion_matrix = confusion_matrix(y_test, y_pred_bin)
confusion_matrix


# In[15]:


sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity : ', specificity)


# In[16]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for White Win classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[17]:


auc(fpr, tpr)


# In[18]:


#perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
#eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[19]:


#explainer = shap.TreeExplainer(model)
#shap_values = explainer.shap_values(X_test)

#shap.summary_plot(shap_values[1], X_test, plot_type="bar")


# In[20]:


#shap.summary_plot(shap_values[1], X_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




