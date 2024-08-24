import pandas as pd
import os
import glob

input_csv_pattern = 'C:/Users/adminroot/AppData/LocalLow/HW/MicroGesture/11Joints/*/*/*.csv'
for csv_file in glob.glob(input_csv_pattern):

        # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 找出需要删除的列名
    columns_to_remove = [col for col in df.columns if
                         any(keyword in col for keyword in ['Proximal', 'Intermediate', 'Palm', 'Metacarpal'])]

    # 删除这些列
    df = df.drop(columns=columns_to_remove)

    # 保存修改后的数据到原路径
    df.to_csv(csv_file, index=False)
    print(f"Columns removed and CSV saved to {csv_file}")

