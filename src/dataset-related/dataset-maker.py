# Since the distributions of samples are not auniform, therefore,
# concatenating the datasets into one and then it will be splitted.


import polars

train = polars.read_csv('datasets/train.csv')
val = polars.read_csv('datasets/validation.csv')
test = polars.read_csv('datasets/test.csv')

dataset = train.vstack(val).vstack(test)

print(dataset.shape)

dataset.write_csv('datasets/News dataset.csv')
print('Dataset saved !')