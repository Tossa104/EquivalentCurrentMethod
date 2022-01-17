import math
import numpy as np
import my_function_v2
from tqdm import tqdm
import time

if __name__ == '__main__':
    p = np.array([0,0.1,0.50])
    a = 0.02
    I = 20
    w = 0.03
    d = 0.03
    h = 0.03
    cen = np.array([0,0.1,0.15])
    B = my_function_v2.B_solver(cen, w, d, h, 1, 1, 1, a, I, 1/14872, p, True, True)
    print(B)


####################################################################
#   csvで座標のリストを当てると，対応する磁束密度のリストができる
###################################################################
    # p_list = np.loadtxt("./p_list.csv", delimiter=",", encoding='utf-8')
    # # print(p_list)
    # n = p_list.shape[0]
    # # print(n)
    # B_list = []

    # for i in range(n):
    #     p_ = p_list[i]
    #     # print("p_",p_)
    #     B_ = my_function_v2.B_solver(cen, w, d, h, 1, 1, 1, a, I, 1/14872, p_, False, False)
    #     B_list.append(B_[0])
    # B_list_np = np.array(B_list)
    # B_list_np.reshape(p_list.shape)
    # print(B_list_np)
    # np.savetxt('B_iron_.csv' ,B_list_np , delimiter=",")
