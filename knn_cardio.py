import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# inisialisasi
cardio = pd.read_excel('cardio_train.xlsx')
x = cardio.iloc[:, :8].values
y = cardio['CLASS cardio'].values

# memisahkan data set dengan data training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# deklarasi fungsi menghitung jarak
def distance(pa, pb):
    # Euclidean
    return sum((pa-pb)**2)**0.5
    # Manhattan
    # return sum(abs(pa-pb))

# deklarasi fungsi KNN
def KNN(x, y, x_query, k):

    # inisialisasi untuk menghitung jarak setiap data training
    m = x.shape[0]
    distances = []
    
    # menghitung jarak setiap data training
    for i in range(m):
        dis = distance(x_query, x[i])
        distances.append((dis, y[i]))
    
    # mengurutkan jarak
    distances = sorted(distances)
    
    # mengambil sebanyak k data training
    distances = distances[:k]
    distances = np.array(distances)
    labels = distances[:, 1]
    
    # menentukan kelas data test
    uniq_label, counts = np.unique(labels, return_counts=True)
    pred = uniq_label[counts.argmax()]
    
    return (pred)
    
k = 9
pred = KNN(x_train, y_train, x_test[0], k)
print('Class Cardio:', pred)

# inisialisasi menghitung akurasi
prediction = []
for i in range(100):
    p = KNN(x_train, y_train, x_test[i], k)
    prediction.append(p)

# menghitung akurasi
predictions = np.array(prediction)
percentage = (y_test[:100] == predictions).sum()/len(predictions)
print('Akurasi:', percentage)


# perbandingan menggunakan sklearn
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train, y_train)

pred = knn.predict(x_test[0].reshape(1, -1))
print('Class Cardio sklearn:', pred)

prediction = []
for i in range(100):
    p = knn.predict(x_test[i].reshape(1, -1))
    prediction.append(p)
percent = (y_test[:100] == prediction).sum()/len(prediction)
print('Akurasi sklearn:', percent)