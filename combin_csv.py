import pandas as pd
import glob
import os

def myFunc(e):
    file="/home/naamah/Downloads/"
    if e==file+"f1.csv":
        return 1
    elif e==file+"f2.csv":
        return 2
    elif e==file+"f3.csv":
        return 3
    elif e==file+"f4.csv":
        return 4
    elif e==file+"f5.csv":
        return 5
    elif e==file+"f6.csv":
        return 6
    elif e==file+"f7.csv":
        return 7
    elif e==file+"f8.csv":
        return 8
    elif e==file+"f9.csv":
        return 9
    elif e==file+"f10.csv":
        return 10
    elif e==file+"f11.csv":
        return 11
    elif e==file+"f12.csv":
        return 12
    elif e==file+"f13.csv":
        return 13
    elif e==file+"f14.csv":
        return 14
    elif e==file+"f15.csv":
        return 15
    elif e==file+"f16.csv":
        return 16
    elif e==file+"f17.csv":
        return 17
    elif e==file+"f18.csv":
        return 18
    elif e==file+"f19.csv":
        return 19
    elif e == file+"f20.csv":
        return 20
    elif e == file+"f21.csv":
        return 21
    elif e == file+"f22.csv":
        return 22
    elif e == file+"f23.csv":
        return 23


writer = pd.ExcelWriter('/home/naamah/Downloads/GA_Vanilla_d16.xlsx', engine='xlsxwriter')
csv_dir = "/home/naamah/Downloads/"

list = glob.glob(os.path.join(csv_dir, "*.csv"))
list.sort(key=myFunc)
for f in list:
        print(f)
        df = pd.read_csv(f)
        df.to_excel(writer, sheet_name=os.path.basename(f)[:31])

writer.save()




