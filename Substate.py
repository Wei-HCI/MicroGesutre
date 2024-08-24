import pandas as pd
import glob
import os

import glob

# 定义文件路径模式，匹配所有 RightHand 的 CSV 文件
input_csv_pattern = 'C:/Users/adminroot/AppData/LocalLow/HW/MicroGesture/11Joints/*/*/*.csv'

# 使用 glob 匹配所有符合条件的文件
all_csv_files = glob.glob(input_csv_pattern)

# 过滤掉路径中包含 "slide" 的文件
non_slide_csv_files = [f for f in all_csv_files if 'slide' not in f]


    # if 'Value' in df.columns:
    #     df = df.drop(columns=['Value'])
    # #创建一个新的列'state'，初始值为4，长度与数据框的行数相同
    # if 'state' in df.columns:
    #     df['state'] = 4
    # df.to_csv(csv_file, index=False)
    # print(f'Updated {csv_file}')

# # 遍历所有符合条件的CSV文件
# for csv_file in glob.glob(input_csv_pattern):
#     #读取CSV文件
#     df = pd.read_csv(csv_file)
#     if 'Value' in df.columns:
#         df = df.drop(columns=['Value'])
#     #创建一个新的列'state'，初始值为4，长度与数据框的行数相同
#     if 'state' in df.columns:
#         df['state'] = 4
#
# 输出符合条件的文件列表
for csv_file in non_slide_csv_files:
    df = pd.read_csv(csv_file)
    # if 'state' not in df.columns:
    #     df['state'] = 4
    #处理'Value'列的数值并更新'state'列
    # if 'Value' in df.columns:
    #     df.loc[(df['Value'] > 0) & (df['Value'] <= 0.25), 'state'] = 3
    #     df.loc[(df['Value'] > 0.25) & (df['Value'] <= 0.5), 'state'] = 2
    #     df.loc[(df['Value'] > 0.5) & (df['Value'] <= 0.75), 'state'] = 1
    #     df.loc[(df['Value'] > 0.75) & (df['Value'] <= 1), 'state'] = 0
    if 'Value' in df.columns:
        df = df.drop(columns=['Value'])
    if 'state' not in df.columns:
        df['state'] = 4
    df.to_csv(csv_file, index=False)
    print(f'Updated {csv_file}')
#
#
#
