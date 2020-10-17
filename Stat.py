#！usr/bin/env python
#encoding:utf-8
'''
__Author__:Lsz
Function:
'''
import random
import numpy as np
from collections import defaultdict
class Stat_data():
    def __init__(self,ipath,opath):
        self.ipath = ipath
        self.opath = opath
        self.name_all = []
        self.name_set = []
        self.val = []
        self.max= 0
        self.org_data = []
        self.split_value = []
        self.distance = 2000
        self.name_val_dic = defaultdict(list)
        self.ofile = f = open(self.opath,'w')
    def read_file(self):
        f = open(self.ipath,'r')
        for line in f.readlines()[1::]:
            print(line)
            self.org_data.append(line)
            line = line.split()[0:2]
            print(line)
            self.name_all.append(line[0])
            self.val.append(int(line[1]))
        self.name_set = list(set(self.name_all))
    def calc_spilt_val(self):
        self.max = np.max(self.val)
        num = self.max  // self.distance
        for i in range(num+1):
            self.split_value.append(i * self.distance)
        self.split_value.append(self.max)
    def process_data(self):
        for i in range(len(self.name_all)):
            name = self.name_all[i]
            value = self.val[i]
            self.name_val_dic[name].append(value)
        for n in self.name_val_dic.keys():
            tmp_dic = defaultdict(list)
            for sp in self.split_value:
                tmp_dic[sp] = 0
            for val in self.name_val_dic[n]:
                for i in range(len(self.split_value)-1):
                    tmp_0 = self.split_value[i]
                    tmp_1 = self.split_value[i+1]
                    if tmp_0 < val <= tmp_1:
                        tmp_dic[tmp_0] += 1
            self.write_dic(n,tmp_dic)
        self.ofile.close()
    def write_dic(self,name,dic):
        for key in dic.keys():
            if dic[key] > 0:
                key_1 = key + 2000
                if key_1 > self.max:
                    key_1 = self.max
                string = name + ":" + "[" + str(key) + " "+ str(key_1) + "]" + "->"+ str(dic[key])
                self.ofile.write(string)
                self.ofile.write('\n')
    def write_shift(self):
        f = open(self.ipath+"_mydata.txt",'w')
        for line in self.org_data:
            line = line.split(',')
            line[0] = 'NC_000' + str(random.randint(0,23))
            line = ','.join(line)
            print(line)
            f.write(line)
            # f.write('\n')
        f.close()
if __name__ == '__main__':
    ipath = r'C:\Users\Administrator\Desktop\Bacillus_subtilis.snp.vcf'
    opath = r'C:\Users\Administrator\Desktop\data_process.txt'
    data_process = Stat_data(ipath,opath)
    data_process.read_file()
    # data_process.write_shift()
    data_process.calc_spilt_val()
    data_process.process_data()
