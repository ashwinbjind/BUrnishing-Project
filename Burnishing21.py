#!/usr/bin/env python
# coding: utf-8

# ### Prediction of surface roughness values after ROLLER BURNISHING PROCESS on CYLENDRICAL SURFACE ON ALUMINIUM WORK PIECES.
# In a single roller burnishing process, a hardened and polished roller is penetrated against a revolving cylindrical
# workp
# iece. Rotation of tool is parallel to the axis of rotation of the workpiece. Roller burnishing process is a superior
# chipless finishing process. It is done on machined or ground surfaces for both external and internal surfaces. Roller
# burnishing is used to improve the mechanical properties of surfaces as well as their surface finish.
# In this experiment cylindrical Aluminum alloy (Al-6061) workpiece has been burnished using different machining
# parameters [Speed, Feed, and Depth of Cut and Number of passes.].The experiment was carried on lathe
# machine. It is observed that Depth of Penetration is the significant parameter to control the surface finish (Ra value).
# The AVOVA results are validated using the AHP method. Here regression model also derived and presented. At the
# end the confirmation tests conducted to confirm results with the actual Ra values.

# ### SPEED in rpm
# ### Feed     in mm/rev.
# ### DOC   in mm.
# ### NOP   int number.
# 
# ### Surface Roughness Value (Ra)

# In[ ]:





# In[1]:


import pandas   as pd
import numpy    as np
import seaborn  as sns

from pandas_profiling        import ProfileReport
from sklearn.linear_model    import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score,confusion_matrix


# In[2]:


df=pd.read_csv("burnishing12.csv")
df.head()


# In[3]:


#data=(df.drop("Unnamed: 6",axis=1))


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


median=df.median()
median


# In[7]:


meadian_NOP=df['NOP'].median()
meadian_DOC=df['DOC'].median()
meadian_Speed=df['Speed'].median()
df['DOC']=df['DOC'].fillna(meadian_DOC)
df['NOP']=df['NOP'].fillna(meadian_NOP)
df['Speed']=df['Speed'].fillna(meadian_Speed)


# In[8]:


df.isnull().sum()


# In[9]:


from collections import Counter


# In[10]:


Counter(df.Speed)


# In[11]:


Counter(df.DOC)


# In[12]:


Counter(df.NOP)


# In[13]:


sns.boxplot(x=df['Speed'],color="pink")


# In[14]:


sns.boxplot(x=df['Feed'],color="green")


# In[15]:


sns.boxplot(x=df['DOC'],color="yellow")


# In[16]:


sns.distplot(df['Ra'])


# In[17]:


profile = ProfileReport(df, title="Pandas Profiling Report")
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
profile


# In[18]:


X = df.iloc[:,0:4] #independent columns
y = df.iloc[:,-1]


# In[19]:


X.shape


# In[20]:


y.shape


# In[21]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=13,test_size=0.2)
X_train.shape


# In[22]:


X_test.shape


# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[24]:


model=LinearRegression()
model.fit(X_train,y_train) 
y_predict=model.predict(X_test)
r2_score(y_test,y_predict)


# In[25]:


print(y_predict)


# In[26]:


import pickle 
pickle_out = open("BURNISHING21.pkl", mode = "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()


# In[27]:


get_ipython().system('pip install -q pyngrok')


# In[29]:


get_ipython().system('pip install -q streamlit')


# In[ ]:


get_ipython().system('pip install -q streamlit_ace')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'app.py', ' \nimport pickle\nimport streamlit as st\n \n# loading the trained model\npickle_in = open(\'BURNISHING21.pkl\', \'rb\') \nclassifier = pickle.load(pickle_in)\n \n@st.cache()\n  \n# defining the function which will make the prediction using the data which the user inputs \ndef prediction(Speed,Feed,DOC,NOP):   \n \n    # Pre-processing user input    \n    if Gender == "Male":\n        Gender = 0\n    else:\n        Gender = 1\n \n    if Married == "Unmarried":\n        Married = 0\n    else:\n        Married = 1\n \n    if Credit_History == "Unclear Debts":\n        Credit_History = 0\n    else:\n        Credit_History = 1  \n \n    LoanAmount = LoanAmount / 1000\n \n    # Making predictions \n    prediction = classifier.predict( \n        [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])\n     \n    if prediction == 0:\n        pred = \'Rejected\'\n    else:\n        pred = \'Approved\'\n    return pred\n      \n  \n# this is the main function in which we define our webpage  \ndef main():       \n    # front end elements of the web page \n    html_temp = """ \n    <div style ="background-color:yellow;padding:13px"> \n    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> \n    </div> \n    """\n      \n    # display the front end aspect\n    st.markdown(html_temp, unsafe_allow_html = True) \n      \n    # following lines create boxes in which user can enter data required to make prediction \n    Gender = st.selectbox(\'Gender\',("Male","Female"))\n    Married = st.selectbox(\'Marital Status\',("Unmarried","Married")) \n    ApplicantIncome = st.number_input("Applicants monthly income") \n    LoanAmount = st.number_input("Total loan amount")\n    Credit_History = st.selectbox(\'Credit_History\',("Unclear Debts","No Unclear Debts"))\n    result =""\n      \n    # when \'Predict\' is clicked, make the prediction and store it \n    if st.button("Predict"): \n        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) \n        st.success(\'Your loan is {}\'.format(result))\n        print(LoanAmount)\n     \nif __name__==\'__main__\': \n    main()')

