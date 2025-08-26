filename='path to EColiK12.gbff'
record = SeqIO.read(filename, "genbank")
genome_coli=record.seq
filename='path to BacSub.gbff'
record = SeqIO.read(filename, "genbank")
genome_sub=record.seq
filename='path to AE006468.gb'
record = SeqIO.read(filename, "genbank")
genome_nella=record.seq

e_coli_dict={}
for i in range(2000,len(genome_coli)-9-1000):
    key=str(genome_coli[i:i+10])
    if key in e_coli_dict.keys():
        e_coli_dict[key].append(i)
    else:
        e_coli_dict[key]=[i]

b_sub_dict={}
for i in range(2000, len(genome_sub)-9-1000):
    key=str(genome_sub[i:i+10])
    if key in b_sub_dict.keys():
       b_sub_dict[key].append(i)
    else:
        b_sub_dict[key]=[i]


nella_dict={}
for i in range(2000,len(genome_nella)-9-1000):
    key=str(genome_nella[i:i+10])
    if key in nella_dict.keys():
        nella_dict[key].append(i)
    else:
        nella_dict[key]=[i]


int_cs=set(e_coli_dict.keys() & set(b_sub_dict.keys()))
three_inter_set=set(int_cs & nella_dict.keys())

test_set=random.sample(three_inter_set,1000)

output={"Intersection":[],"organism":[], "coordinate":[],"index":[]}
orgs=['coli','sub','nella']
org_dict={'coli':e_coli_dict,'sub':b_sub_dict,'nella':nella_dict}
counter=0
for i in test_set:
    for org in orgs:
        for k in org_dict[org][i]:
            output["Intersection"].append(i)
            output["organism"].append(org)
            output["coordinate"].append(k)
            output["index"].append(counter)
            counter+=1
output=pd.DataFrame(output)
output.to_csv("file.csv")   

ht_meta=pd.read_csv("file.csv")

# confirm that (intersection,coordinate) is unique
listss=[]
for i in range(len(ht_meta)):
    if (ht_meta.iloc[i]["Intersection"],ht_meta.iloc[i]["coordinate"]) in listss:
        print((ht_meta.iloc[i]["Intersection"],ht_meta.iloc[i]["coordinate"]), "error")
    else:
        listss.append((ht_meta.iloc[i]["Intersection"],ht_meta.iloc[i]["coordinate"]))


output={"Intersection":[], "head":[],"tail":[],"index":[]}
counter=0
for i in test_set:
    df=ht_meta[ht_meta["Intersection"]==i]
    for j in range(len(df)):
        for k in range(len(df)):
            output["Intersection"].append(i)
            output["head"].append(df.iloc[j]["coordinate"])
            output["tail"].append(df.iloc[k]["coordinate"])
            output["index"].append(counter)
            counter+=1

output=pd.DataFrame(output)
output.to_csv("file_seq.csv")

