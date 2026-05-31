import shutil
import pandas as pd

shutil.copy('train/train.csv', 'train/train_pre_merge.csv')

new = pd.read_csv('data/new_data.csv')
train = pd.read_csv('train/train.csv')
merged = pd.concat([train, new], ignore_index=True)
merged.to_csv('train/train.csv', index=False)
print(f'Merged {len(new)} new rows into train.csv (total: {len(merged)})')
