from tinysklearn.preprocessing import MinMaxScaler

X = [[1, 2], [3, 4], [5, 6]]

scaler = MinMaxScaler()
print(scaler.fit_transform(X))
