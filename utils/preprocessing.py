import pandas as pd

def read_fasta(path):
    f = open(path, 'r')
    seq_list = []
    name_list = []
    lines = f.readlines()
    for item in lines:
        name_list.append(item[1:].split()[0]) \
            if item[0] == '>' else seq_list.append(item.split()[0])
    df = pd.DataFrame({'name': name_list, 'seq': seq_list})
    return df


p_sAMP = read_fasta('../dataset/Benchmark/sAMPs.fasta')
p_sAMP.insert(2, 'IssAMP', 1)

n_sAMP = read_fasta('../dataset/Benchmark/non-sAMPs.fasta')
n_sAMP.insert(2, 'IssAMP', 0)

sAMP = pd.concat([p_sAMP, n_sAMP])

sAMP.to_csv('../dataset/Benchmark/sAMPs.csv', index=None, header=None)
