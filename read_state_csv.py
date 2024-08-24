import pandas as pd
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import parser
from utils.pre_data import load_data
from utils.skeleton import SkeletonData
from Matrix import initialise_model
from model.rh_net import STADualNet

# 输入CSV文件路径
input_csv_file = 'C:/Users/adminroot/Desktop/RightHandData_Rolling.csv'
# 输出CSV文件路径
output_csv_file = 'C:/Users/adminroot/AppData/LocalLow/HW/MicroGesture/static/Participant_0/scissors/RightHandData_Rolling.csv'
result_file = 'C:/Users/adminroot/Desktop/result.txt'


# 提前进行CUDA初始化
torch.cuda.init()


def custom_softmax(input_tensor):
    """
    自定义Softmax函数
    :param input_tensor: 输入张量 (通常是模型的输出)
    :return: 应用softmax后的张量
    """
    # 计算每个元素的指数值
    exp_tensor = torch.exp(input_tensor - torch.max(input_tensor))

    # 计算归一化系数，即所有元素的总和
    sum_exp = torch.sum(exp_tensor, dim=1, keepdim=True)

    # 计算softmax输出
    softmax_output = exp_tensor / sum_exp

    return softmax_output

# 读取和处理CSV文件的函数
def read_csv_and_process():
    arg = parser.parser_args()
    num_frame = 20

    # 加载CSV数据
    data_loading_start_time = time.time()
    features, labels, states = load_data(output_csv_file, num_frame)
    data_loading_end_time = time.time()
    print(f"数据加载时间: {data_loading_end_time - data_loading_start_time:.6f} 秒")

    features = np.array(features)
    labels = np.array(labels)
    states = np.array(states)

    custom_dataset = SkeletonData(features, labels, states)
    custom_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 初始化模型

    model = initialise_model(arg, STADualNet)

    # 检查是否使用 DataParallel
    is_data_parallel = torch.cuda.device_count() > 1
    if is_data_parallel:
        model = torch.nn.DataParallel(model)

    # 加载模型权重
    pt_model_path = 'C:/Users/adminroot/Downloads/HG_Recognizer-main/220824_RH_model_20f_noise_1-2_256/ep_5_acc_0.979_194730.pt'
    model.module.load_state_dict(torch.load(pt_model_path))


    model.eval()

    results = []
    with torch.no_grad():

        for data in custom_loader:


            features, labels, states = data

            # 转换 features 为张量并移动到GPU上

            features_tensor = features.float().cuda()


            # 进行推理

            outputs = model(features_tensor)


            # 解包输出

            if isinstance(outputs, tuple):
                outputs = outputs[0]


            # 将推理结果进行Softmax

            #softmax_output = custom_softmax(outputs)


            # 将推理结果移动到CPU并转换为NumPy数组

            cls_output = outputs.cpu().numpy()


            results.append(cls_output)
            # 打印每个步骤的运行时间


    # 将结果写入文件
    with open(result_file, 'w') as file:
        for result in results:
            file.write(f'Classification Output: {result}\n')


if __name__ == '__main__':
    while True:
        try:
            # 读取CSV文件
            df = pd.read_csv(input_csv_file)

            # 检查数据框是否为空
            if df.empty:
                print(f"错误: 文件 {input_csv_file} 为空。")
                time.sleep(0.2)
                continue

            # 检查是否存在'Unnamed: 190'列，并将其删除
            # if 'Unnamed: 190':
            #     df = df.drop(columns=['Unnamed: 190'])

            # 创建一个新的列'state'，值为4，长度与数据框的行数相同
            #df['state'] = 4
            columns_to_remove = [col for col in df.columns if
                                 any(keyword in col for keyword in ['Proximal', 'Intermediate', 'Palm', 'Metacarpal'])]

            # 删除这些列
            df = df.drop(columns=columns_to_remove)

            if 'state' not in df.columns:
                df['state'] = 4

            # 保存修改后的数据框到新文件
            df.to_csv(output_csv_file, index=False)

            start_time = time.time()
            read_csv_and_process()
            end_time = time.time()
            print(f"程序运行时间: {end_time - start_time:.2f} 秒")
            time.sleep(0.2)

        except pd.errors.EmptyDataError:
            print(f"错误: 文件 {input_csv_file} 为空。")
            time.sleep(0.2)
        except Exception as e:
            print(f"发生错误: {e}")
            time.sleep(0.2)
