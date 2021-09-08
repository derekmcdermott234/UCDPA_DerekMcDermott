print(sp500_0.shape)
print(sp500_0.head(5))
missing_values_count_0=sp500_0.isnull().sum()
print(missing_values_count_0[:])
null_data_0 = sp500_0[sp500_0.isnull().any(axis=1)]
print(null_data_0)