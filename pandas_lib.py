#！usr/bin/env python
#encoding:utf-8
'''
__Author__:Lsz
Function:
'''
import pandas as pd
class process_data():
    def __init__(self,ipath,opath):
        self.ipath = ipath
        self.opath = opath
        self.pd_in = pd.DataFrame()
        self.pd_out = pd.DataFrame()
        self.value_max = 0
        self.interval = []
        self.distance = 2000
    def read_file(self):
        self.pd_in = pd.read_table(self.ipath).loc[:,["#CHROM","POS"]]
        self.value_max = self.pd_in["POS"].max()
        self.interval = [i for i in range(0,self.value_max,self.distance)]
        self.interval.append(self.value_max)
    def process(self):
        self.pd_out = pd.DataFrame(columns=["name","interval1","interval2","num"])
        print(self.pd_in.head())
        index_cnt = 0
        for name in self.pd_in["#CHROM"].unique():
            df_tmp = self.pd_in.loc()[:][self.pd_in["#CHROM"] == name]
            print(df_tmp.head())
            for i in range(len(self.interval) - 1):
                cnt = 0
                for pos in df_tmp["POS"]:
                    if self.interval[i] < pos <= self.interval[i+1]:
                        cnt += 1
                self.pd_out.loc[index_cnt] = [name,self.interval[i]+1,self.interval[i+1],cnt]
                index_cnt += 1
            self.pd_out.to_excel(self.opath,index=False)
if __name__ == '__main__':
    ipath = r'C:\Users\Administrator\Desktop\Bacillus_subtilis.snp.vcf'
    opath = r'C:\Users\Administrator\Desktop\Bacillus_subtilis.snp.vcf_out.xls'
    proces = process_data(ipath,opath)
    proces.read_file()
    proces.process()
