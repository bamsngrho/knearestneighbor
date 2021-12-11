import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


sensus = {'tinggi' : [167, 183, 155, 177, 180], 'berat' : [52, 60, 89, 49, 77], 'jk': ['pria', 'wanita', 'wanita', 'pria', 'wanita']}
df_sensus= pd.DataFrame(sensus)
df_sensus


fig, ax = plt.subplots()
for jk, d in df_sensus.groupby('jk'):
  ax.scatter(d['tinggi'], d['berat'], label=jk)

plt.title('Selebaran Data Tinggi Badan, Berat Badan dan Jenis Kelamin')
plt.xlabel('Tinggi Badan (cm)')
plt.ylabel('Berat Badan (kg)')
plt.legend()
plt.grid(True)
plt.show()


x_train = np.array(df_sensus[['tinggi', 'berat']])
y_train = np.array(df_sensus['jk'])

print(x_train)
print(y_train)


lb =LabelBinarizer()
y_train = lb.fit_transform(y_train)
print(y_train)

y_train = y_train.flatten()
print(y_train)

from sklearn.neighbors import KNeighborsClassifier
K = 3
model = KNeighborsClassifier(n_neighbors =K)
model.fit(x_train, y_train)

tinggi_badan = 160
berat_badan = 62
x_new = np.array([tinggi_badan, berat_badan]).reshape(1, -1)
print(x_new)

y_new = model.predict(x_new)
lb.inverse_transform(y_new)
y_new

fig, ax = plt.subplots()
for jk, d in df_sensus.groupby('jk'):
  ax.scatter(d['tinggi'], d['berat'], label=jk)

plt.scatter(tinggi_badan, berat_badan, marker='s', color='red', label='mister x')
plt.title('Selebaran Data Tinggi Badan, Berat Badan dan Jenis Kelamin')
plt.xlabel('Tinggi Badan (cm)')
plt.ylabel('Berat Badan (kg)')
plt.legend()
plt.grid(True)
plt.show

mister_x = np.array([tinggi_badan, berat_badan])
mister_x


data_jarak = [euclidean(mister_x, d) for d in x_train]
data_jarak

df_sensus['jarak'] = data_jarak
df_sensus.sort_values(['jarak'])

x_test = np.array([[167, 52], [183, 60], [155, 89], [177, 49], [180, 77]])
y_test = lb.transform(np.array(['wanita', 'pria', 'wanita','wanita','pria'])).flatten()

print(x_test)
print(y_test)

y_pred = model.predict(x_test)
y_pred


acc = accuracy_score(y_test, y_pred)
print(acc)


prec = precision_score(y_test, y_pred)
print(prec)

rec = recall_score(y_test, y_pred)
print(rec)