import datasets
# Download wikihowAll.csv
df = datasets.load_dataset('wikihow', 'all', data_dir='.')

toy_train = df['validation'].to_pandas()

toy_val = df['test'].to_pandas().sample(n=100)

toy_train.reset_index(inplace=True,drop=True)
toy_val.reset_index(inplace=True,drop=True)
toy_train.to_json('../processed/train.json',orient='split')
toy_val.to_json('../processed/val.json',orient='split')
