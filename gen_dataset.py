#!/usr/bin/env python

from ROOT import TTree,TFile

import numpy as np

def gen_data():

    file_index = open('index','r')
    index = {}
    i = 0
    for line in file_index:
        ID = int(line.split()[0])
        index[ID] = i
        i=i+1
    
    f = TFile("final_40kHz_200nsWindow_100.root",'read')
    
    t = f.Get('t')
    datas = []
    labels = []
    entries = t.GetEntries()
    DarkN = 0
    for i in range(entries):
        t.GetEntry(i)
        nhits = t.GetLeaf("nhits").GetValue(0)
        data = []
        nhits_C14 = int((t.GetLeaf("nhits_C14").GetValue(0)))
        if nhits_C14 > 0 and nhits_C14<=15:
            continue
        if nhits_C14 == 0 and DarkN >= 1000:
            continue

        if nhits_C14 == 0 and DarkN < 1000:
            labels.append([0])
            DarkN += 1

        if nhits_C14 > 15:
            labels.append([1])
    
        for k in range(0,17739*4):
            data.append(0)
        for j in range(int(nhits)):
            time = int((t.GetLeaf("hitTime").GetValue(j))-i*200)
            x = int((t.GetLeaf("GlobalPosX").GetValue(j)))
            y = int((t.GetLeaf("GlobalPosY").GetValue(j)))
            z = int((t.GetLeaf("GlobalPosZ").GetValue(j)))
            pmtID = int((t.GetLeaf("pmtID").GetValue(j)))
            if pmtID > 20000:
                continue
            data[int(index[pmtID]*4)] = time
            data[int(index[pmtID]*4+1)] = x
            data[int(index[pmtID]*4+2)] = y
            data[int(index[pmtID]*4+3)] = z
    
        datas.append(data)
    
    return (np.array(datas,'i'),np.array(labels,'i'))

if __name__ == "__main__":
    datas = gen_data()
    file = open('Data','w')
    for i in range(len(datas[0])):
        for j in range(17739*4):
            file.write(str(datas[0][i][j])+'\t')
        file.write('\n'+str(datas[1][i][0])+'\n')
    print(len(datas[0]))
    print("hello")
