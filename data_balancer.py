# script to balancing the t_data as well as the validation data


import os, random

with open('v_data.csv') as fin:
    true  = []
    false = []
    for line in fin.readlines():
        line = line.strip('\n').strip(';')
        
        try:
            if int(line[-1]) == 1:
                true.append(line)
            else:
                false.append(line)
        except:
            continue
            
            
            
print('positiv cases : ', len(true))
print('negativ cases : ', len(false))


if len(true) < len(false):
    all_data = true + false[:len(true)]
    random.shuffle(all_data)
    with open('./balanced_v_data.csv', 'w') as fout:
        for line in all_data:
            fout.write(line + '\n')
    
