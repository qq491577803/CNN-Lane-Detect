import numpy as np
import matplotlib.pyplot as plt
import xlwt
import xlrd
from glob import glob

class kalman:
    def __init__(self,y,Q,R):
        self.y = y
        self.n_inter = len(y)
        self.x = [i for i in range(self.n_inter)]
        self.xwhat = np.zeros((self.n_inter,))
        self.P = np.zeros((self.n_inter,))
        self.xwhatminus = np.zeros((self.n_inter,))
        self.Pminus = np.zeros((self.n_inter,))
        self.K = np.zeros((self.n_inter,))
        # self.R = 0.1**2
        # self.Q = 1e-3
        self.R = R
        self.Q = Q
        self.xwhat[0] = y[0]
        self.P[0] = 1.0
    def filter(self):
        for k in range(1,self.n_inter):
            self.xwhatminus[k] = self.xwhat[k-1]
            self.Pminus[k] = self.P[k-1] + self.Q
            self.K[k] = self.Pminus[k] / (self.Pminus[k] + self.R)
            self.xwhat[k] = self.xwhatminus[k] + self.K[k] * (self.y[k] - self.xwhatminus[k])
            self.P[k] = (1-self.K[k])*self.Pminus[k]
        return self.xwhat

def read(path):
    wb = xlrd.open_workbook(path)
    sheet = wb.sheet_by_index(0)
    ncols = sheet.ncols
    data = []
    for i in range(ncols):
        tmp = []
        for j in sheet.col_values(i):
            if j != "":
                tmp.append(j)
        data.append(tmp)
    return data
if __name__ == '__main__':
    path = './mydata/mydata.xlsx'
    data = read(path)
    wb = xlwt.Workbook()
    table = wb.add_sheet('res',cell_overwrite_ok=True)
    cnt = 0
    for col in data:
        q,r = 0.002,0.01
        if len(col) == 0:
            continue
        y = kalman(col[2::], q, r).filter()
        print(col[2::])
        for row in range(len(y)):
            table.write(row,cnt,round(y[row],0))
        cnt += 1
        x = [i for i in range(len(y))]
        title = 'r:' + str(r) + 'q:' + str(q)
        plt.title(title)
        plt.plot(x, col[2::])
        plt.plot(x, y)

        plt.show()
    wb.save('./mydata/res.xls')