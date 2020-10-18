import xlrd
import xlwt
from glob import glob
flag = input("Please click enter to start !")
print("Please wait ...")

files = glob("*.xlsm")
files.sort()
f = xlwt.Workbook()
table = f.add_sheet(u'combine',cell_overwrite_ok=True)
#build head
headfile = files[0]
headbook = xlrd.open_workbook(headfile)
headsheet = headbook.sheet_by_index(0)
headname = headsheet.row_values(5)
rows = headname
n = 0
table.write(0,0,'PointName')
for i in range(len(headname)):
    table.write(0, i+1, headname[i])
    n=n+1
#build content
row = 1
if flag !='':
    for file in files:
        # print(file)
        pointname = file.split('.')[0]
        print(pointname)
        table.write(row,0,pointname)
        workbook = xlrd.open_workbook(file)
        sheet = workbook.sheet_by_index(0)
        row6 = sheet.row_values(6)
        row7 = sheet.row_values(7)
        row8 = sheet.row_values(8)
        assert len(row6) == len(rows)
        assert len(row7) == len(rows)
        assert len(row8) == len(rows)
        for i in range(len(row6)):
            table.write(row,i+1,row6[i])
        row = row + 1
        for i in range(len(row7)):
            table.write(row, i + 1, row7[i])
        row = row + 1
        for i in range(len(row8)):
            table.write(row, i + 1, row8[i])
        row = row + 1
        print(row6)

        for i in range(len(row8)):
            table.write(row,0,'Average')
            if headname[i] == 'Comment' or headname[i] == 'Warnings':
                aver_value = ''
            else:
                if row6[i]=='' and row7[i]=='' and row8[i]=='':
                    aver_value = ''
                elif row6[i]=='' or  row7[i]=='' or row8[i]=='':
                    if row6[i] == '': row6[i] = 0
                    if row7[i] == '': row7[i] = 0
                    if row8[i] == '': row8[i] = 0
                    aver_value = (float(row6[i]) + float(row7[i]) + float(row8[i])) / 3
                else:
                    aver_value = (float(row6[i]) + float(row7[i]) + float(row8[i])) / 3
                table.write(row,i+1,aver_value)
        row = row + 1
    print(pointname + 'has been completed !')
else:
    for file in files:
        # print(file)
        pointname = file.split('.')[0]
        print(pointname)
        table.write(row,0,pointname)
        workbook = xlrd.open_workbook(file)
        sheet = workbook.sheet_by_index(0)
        row6 = sheet.row_values(6)
        row7 = sheet.row_values(7)
        row8 = sheet.row_values(8)
        assert len(row6)==len(rows)
        assert len(row7) == len(rows)
        assert len(row8) == len(rows)
        for i in range(len(row8)):
            # table.write(row,0,'Average')
            if headname[i] == 'Comment' or headname[i] == 'Warnings':
                aver_value = ''
            else:
                if row6[i]=='' and row7[i]=='' and row8[i]=='':
                    aver_value = ''
                elif row6[i]=='' or  row7[i]=='' or row8[i]=='':
                    if row6[i] == '': row6[i] = 0
                    if row7[i] == '': row7[i] = 0
                    if row8[i] == '': row8[i] = 0
                    aver_value = (float(row6[i]) + float(row7[i]) + float(row8[i])) / 3
                else:
                    aver_value = (float(row6[i]) + float(row7[i]) + float(row8[i])) / 3
                table.write(row,i+1,aver_value)
        row = row + 1
    print(pointname + 'has been completed !')
print('All the file has been processed !')
f.save('./DMS500_Data_combine.xls')
print("The combined file is in the same root of datafile !")
print('The file named DMS500_Data_combine.xls')
input("Please click enter to end  !")