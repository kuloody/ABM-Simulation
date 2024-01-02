# Save Model Using Pickle
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
#Data retrieval

df = pd.read_csv('/home/zsrj52/Downloads/SimClass/dataset/OldPIPSNullRemoved.csv')
df=df.dropna(subset=['FSM'])
df=df.dropna(subset=['End_age'])
df=df.dropna(subset=['Start_Vocabulary'])
df=df.dropna(subset=['Start_Reading'])
df=df.dropna(subset=['End_Reading'])
df=df.dropna(subset=['IDACI'])
df=df.dropna(subset=['Start_age'])
df=df.dropna(subset=['Inattentiveness'])
df.to_csv('/home/zsrj52/Downloads/SimClass/dataset/OldPIPSNullRemovedallfeatures.csv')

df.describe()

dataframe = df[['Start_maths', 'Start_Reading', 'Start_Vocabulary','FSM','IDACI','Hyperactivity','Impulsiveness', 'Inattentiveness', 'End_age','End_maths']]
array = dataframe.values
X = array[:,0:9]
print('X is:',X)
Y = array[:,9]
print('Y is:',Y)
test_size = 0.33
seed = 7

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = '/home/zsrj52/Downloads/SimClass/LinearModel-9features.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
test = X_test[1].reshape((1, -1))
pred = loaded_model.predict(test)
print(pred)
#Load test object
df = pd.read_csv('/home/zsrj52/Downloads/SimClass/dataset/OldPIPS-SAMPLE.csv')
print(df.iloc[4])
df.drop(columns=['ID','Gender','End_maths','School','Class','End_Reading'], inplace=True)
print(df.iloc[4])
pred = loaded_model.predict(df.iloc[4].values.reshape(1, -1))
print(pred)

