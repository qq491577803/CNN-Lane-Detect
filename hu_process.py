#！usr/bin/env python
#encoding:utf-8
'''
__Author__:Lsz
Function:
'''
import xlrd
import xlwt
from glob import glob

if __name__ == '__main__':
    fns = glob('./*.csv')
    print(fns)