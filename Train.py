import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from keras.preprocessing.sequence import pad_sequences

data_dict = pickle.load(open(r'C:\Users\dc\OneDrive\Desktop\data\data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

# Pad sequences to a maximum length (adjust maxlen according to your data)
maxlen = 200
data = pad_sequences(data, maxlen=maxlen, padding='post', truncating='post', dtype='float32')

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Creating the RandomForestClassifier model
model = RandomForestClassifier()

# Fitting the model on the training data
model.fit(X_train, y_train)

# Predicting the labels on the test data
y_predict = model.predict(X_test)

# Calculating the accuracy of the model
score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% of samples were classified correctly!')



f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()