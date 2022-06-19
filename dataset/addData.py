import random

import pandas as pd

name = []
seq = []
with open(r'F:\data\dbAMP\dbAMPv2.0\dbAMP2_validated.fasta', 'r') as f:
    lines = f.readlines()
    for line in lines:
        name.append(line[1:].split()[0]) if line[0] == '>' else seq.append(line.split()[0])
len_seq = [len(i) for i in seq]
df = pd.DataFrame({'seq': seq, 'len': len_seq, 'interaction': 1})
df = df[df['len'] <= 50]
# df.to_csv('AMPDB/pos.csv', index=None)
len_seq = df['len'][1:].to_list()
hiv = pd.read_table(r'F:\data\transppi\sample\HIV\pro_seq.txt', names=['name', 'seq'])['seq'].to_list()[0:len(len_seq)]

neg = []
for i in range(len(len_seq)):
    r = random.randint(0, 100)
    if len(hiv[i]) < r+len_seq[i]:
        r = 0
    neg.append(hiv[i][r:r+len_seq[i]])
neg_df = pd.DataFrame({"seq": neg, 'interaction': 0})
df = df[['seq', 'interaction']]
pd.concat([df, neg_df]).to_csv('AMPDB/amp.csv', index=None)
