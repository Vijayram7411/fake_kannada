import pandas as pd

# Let's examine the raw files first
print("=== FAKE FILE SAMPLE ===")
with open('data/train_fake.csv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        print(f"Line {i+1}: {line.strip()}")
        if i >= 10:
            break

print("\n=== REAL FILE SAMPLE ===")
with open('data/train_real.csv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        print(f"Line {i+1}: {line.strip()}")
        if i >= 10:
            break

print("\n=== TRYING DIFFERENT PARSING ===")

# Try different parsing approaches
fake_df1 = pd.read_csv('data/train_fake.csv', sep='|', names=['line_num', 'data'], encoding='utf-8')
print(f"Method 1 - Fake rows: {len(fake_df1)}")
print(fake_df1.head())

print("\n=== With header skip ===")
fake_df2 = pd.read_csv('data/train_fake.csv', sep='|', names=['line_num', 'data'], encoding='utf-8', skiprows=1)
print(f"Method 2 - Fake rows: {len(fake_df2)}")
print(fake_df2.head())

print("\n=== Check total lines ===")
with open('data/train_fake.csv', 'r', encoding='utf-8') as f:
    fake_lines = len(f.readlines())

with open('data/train_real.csv', 'r', encoding='utf-8') as f:
    real_lines = len(f.readlines())
    
print(f"Total lines in fake file: {fake_lines}")
print(f"Total lines in real file: {real_lines}")