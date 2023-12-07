#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Data
data = PatientID, Readmission, StaffSatisfaction, CleanlinessSatisfaction,
FoodSatisfaction, ComfortSatisfaction, CommunicationSatisfaction
101, 1, 4, 5, 3, 4, 5
102, 0, 3, 2, 4, 2, 3
103, 1, 5, 4, 5, 4, 4
104, 0, 5, 3, 5, 4, 5
105, 1, 4, 5, 3, 4, 4
106, 1, 2, 3, 2, 3, 3
107, 0, 3, 2, 3, 4, 3
108, 0, 5, 5, 4, 5, 5
109, 1, 4, 4, 4, 4, 4
110, 0, 3, 3, 3, 4, 3
111, 1, 4, 5, 4, 4, 5
112, 0, 2, 3, 2, 3, 2
113, 1, 3, 4, 3, 4, 3
114, 1, 3, 3, 3, 2, 3
115, 0, 5, 4, 5, 5, 5
116, 1, 4, 3, 4, 4, 4
117, 0, 2, 2, 2, 3, 3
118, 0, 5, 5, 4, 4, 5
119, 1, 4, 4, 4, 4, 4
120, 0, 3, 3, 3, 4, 3
121, 1, 4, 5, 5, 4, 4
122, 1, 3, 4, 4, 4, 3
123, 0, 4, 3, 4, 3, 4
124, 0, 2, 2, 2, 3, 3
125, 1, 3, 4, 3, 4, 3
126, 0, 4, 5, 5, 5, 5
127, 1, 3, 4, 4, 4, 3
128, 0, 4, 3, 4, 3, 4
129, 0, 2, 2, 2, 3, 3
130, 1, 3, 4, 3, 4, 3


# In[3]:


# Create a DataFrame from the Data
df = pd.read_csv(pd.compat.StringIO(data))


# In[4]:


# Task 1: Number of ptients who were readmitted
readmitted_patients = df[df['Readmission'] == 1]
num_readmitted = len(readmitted_patients)
print(f"Number of patients who were readmitted: {num_readmitted}")


# In[5]:


# Task 2: Average satisfaction score for each category
satisfaction_columns = ['StaffSatisfaction', 'CleanlinessSatisfaction', 'FoodSatisfaction', 'ComfortSatisfaction', 'CommunicationSatisfaction']
avg_satisfaction = df[satisfaction_columns].mean()
print("\nAverage satisfaction scores:")
print(avg_satisfaction)


# In[6]:


# Task 3: Logistic Regression
X = df[satisfaction_columns]
y = df['Readmission']


# In[7]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[8]:


# Create a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[9]:


# Task 4: Display logistic regression results
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy}")


# In[ ]:


# Task 5: Plot the data points along with the logistic regression curve
plt.figure(figsize=(8, 6))
plt.scatter(X_test['ComfortSatisfaction'], y_test, color='black', label='Actual Data Points')
plt.scatter(X_test['ComfortSatisfaction'], y_pred, color='red', marker='x', label='Predicted Data Points')
plt.xlabel('Comfort Satisfaction')
plt.ylabel('Readmission')
plt.title('Logistic Regression and Data Points')
plt.legend()
plt.show()

