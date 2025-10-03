import pandas as pd
from pathlib import Path

# Load original processed dataset
base = pd.read_csv('data/processed_dataset.csv')
base = base[['text','label']].dropna()

# Load supplements
supp_fake = pd.read_csv('data/supplement.csv')
supp_real = pd.read_csv('data/supplement_real.csv', header=None, names=['text','label'])

# Combine
aug = pd.concat([base, supp_fake, supp_real], ignore_index=True)
aug = aug.dropna().drop_duplicates(subset=['text'])

# Save
Path('data').mkdir(exist_ok=True)
aug.to_csv('data/augmented_dataset.csv', index=False, encoding='utf-8')
print('Saved data/augmented_dataset.csv', aug.shape)
