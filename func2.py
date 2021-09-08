print(sp500_1.shape)
print(sp500_1.head(5))
missing_values_count_1=sp500_1.isnull().sum()
print(missing_values_count_1[:])
null_data_1 = sp500_1[sp500_1.isnull().any(axis=1)]
print(null_data_1)