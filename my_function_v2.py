import numpy as np
import math
from tqdm import tqdm


# Gの式を計算したやつに置き換えた

###############################################

# p : コイルが発生させる磁束密度を求める座標(m) 1×3
# a : コイルの半径(m)
# i : コイルに流れる電流(A)
# [0, 2π]をいくつの区間に分割して区分求積するか
# θ(theta)：原点から磁束密度を求める点を結ぶ直線とZ軸のなす角．単位はrad(ラジアン)
# gem_chap08参照

################################################

def function_r(r, a, theta, phi):
    return (a * np.cos(theta)) / ((r*r + a*a - 2*r*a*np.sin(theta)*np.sin(phi))**(3/2))

def function_theta(r, a, theta, phi):
    return (r * np.sin(phi) - a* np.sin(theta)) / ((r*r + a*a - 2*r*a*np.sin(theta)*np.sin(phi))**(3/2))

def function_x(r, a, theta, phi):
    return (r * np.cos(theta) * np.cos(phi)) / ((r*r + a*a - 2*r*a*np.sin(theta)*np.sin(phi))**(3/2))

# ある点pでの磁束密度を求める関数
def Bc_solver(p : np.ndarray, a, i, n):
    mu0 = (4*np.pi)/10000000
    er = 0
    eth = 0
    ex = 0
    dphi = (2*np.pi) / n
    r = np.linalg.norm(p)
    theta = np.arccos(p[2] / r)
    # if p[0] == 0:
    #     phi = np.pi / 2
    # else:
    #     phi = np.arccos(p[0] / ((p[0]**2 + p[1]**2)**0.5))
    phi = atan_2pi(p[1], p[0])
    # for j in tqdm(range(n)):
    for j in range(n):
        # r方向成分について微小区間を長方形近似し，足し合わせる
        er += function_r(r, a, theta, dphi * j) * dphi

        # θ方向成分について微小区間を長方形近似し，足し合わせる
        eth += function_theta(r, a, theta, dphi * j) * dphi

        # x方向成分について微小区間を長方形近似し，足し合わせる
        ex += function_x(r, a, theta, dphi * j) * dphi
    
    # print("I [A]", i)
    # print("μ_0",mu0)er
    # print("radius [m]",a)
    # print("r [m]",r)
    # print(p[0])
    # print((p[0]**2 + p[1]**2)**0.5)
    # print("θ [rad]",theta)
    # print("phi", phi)
    er = (er * mu0 * i * a) /(4*np.pi)
    eth = (eth * mu0 * i * a) /(4*np.pi)
    ex = (ex * mu0 * i * a) /(4*np.pi)
    # print(er)
    # print(eth)
    # print(ex)
    Bx = er * np.sin(theta) * np.cos(phi) + eth * np.cos(theta) * np.cos(phi)
    By = er *np.sin(theta) *np.sin(phi) + eth * np.cos(theta) * np.sin(phi)
    Bz = er * np.cos(theta) - eth * np.sin(theta)
    return Bx, By, Bz



###################################################

# コイルが磁性体要素の中心に作る磁界の行列(3n × 1)
# [Bx0, By0, Bz0, Bx1,By1, Bz1, ...   ... , Bxn, Byn, Bzn]
# cens : 要素中心のリスト(n×3)
# a : コイル半径(m)
# i : 電流(A)
# n : [0, 2π]をいくつの区間に分割して区分求積するか

##################################################


def vector_Bc( cens :np.ndarray, a, i, n):
    Bc =[]
    # print("Number of elments is ...."," ",cens.shape[0])
    for j in range(cens.shape[0]):
        cen = cens[j]
        Bx, By, Bz = Bc_solver(cen, a, i, n)
        Bc.append(Bx)
        Bc.append(By)
        Bc.append(Bz)
    Bc = np.reshape(Bc, (cens.size, 1))
    # print(Bc.reshape(cens.shape))
    return Bc



#############################################
# 奇数の分割に対応指せるべし################################################################################
# 磁性体を小要素に分割する
# yz平面基準
# cen : 金属の中心座標(m)　1×3
# w : 金属の横幅＝x方向の長さ(m)
# d : 金属の奥行き＝y方向の長さ(m)
# h : 金属の高さ＝z方向の長さ(m)
# n : wの分割数  *偶数のみ
# m : dの分割数  *偶数のみ　
# l : hの分割数  *偶数のみ
# a, b, c :直方体要素の各辺の半分の長さ(m)
# cen_list_np : 要素の中心の座標の配列(m) n×3

#############################################

# 金属の寸法と中心座標と各辺の分割数(偶数)を与えると要素の中心の配列とa,b,cを返す関数
def make_elements( cen : np.ndarray, w, d, h, n, m, l):
    a = w/(2*n)
    b = d/(2*m)
    c = h/(2*l)
    cen_ox, cen_oy, cen_oz = cen[0], cen[1], cen[2]
    # 要素1の中心座標設定
    cen_x, cen_y, cen_z = cen_ox + (n-1)*a, cen_oy + (m-1)*b, cen_oz + (l-1)*c
    cen_list = []
    for i in range(n):
        for j in range(m):
            for k in range(l):
                cen_ix = cen_x - i*2*a
                cen_jy = cen_y - j*2*b
                cen_kz = cen_z - k*2*c
                cen_list.append([ cen_ix, cen_jy, cen_kz])
    cen__list_np = np.array(cen_list)
    # print(cen__list_np)
    return cen__list_np, a, b, c

# def make_elements( cen : np.ndarray, w, d, h, n, m, l):
#     a = w/(2*n)
#     b = d/(2*m)
#     c = h/(2*l)
#     cen_ox, cen_oy, cen_oz = cen[0], cen[1], cen[2]
#     # 要素1の中心座標設定
#     cen_x, cen_y, cen_z = cen_ox + (n-1)*a, cen_oy + (m-1)*b, cen_oz + (l-1)*c
#     cen_list = []
#     for i in range(n):
#         for j in range(m):
#             for k in range(l):
#                 cen_ix = cen_x - i*2*a
#                 cen_jy = cen_y - j*2*b
#                 cen_kz = cen_z - k*2*c
#                 cen_list.append([ cen_ix, cen_jy, cen_kz])
#     cen__list_np = np.array(cen_list)
#     print(cen__list_np)
#     return cen__list_np, a, b, c


# ここまでおｋ
#######################################################################################################################
######################################################################

# 等価電流法用関数

######################################################################

######################################

# u, v, w :磁束密度を求めたい点Pの座標(m)
# a, b, c :直方体要素の各辺の半分の長さ(m)
# p：要素の中心の位置ベクトル(m)
# ~_p : 動かない座標(デフォルトはx成分)が要素中心から正の位置のときの関数
# ~_m : 動かない座標(デフォルトはx成分)が要素中心から負の位置のときの関数

######################################

########################################################################

# most stupid way
# 磁性体の中心がz軸上の時で確認　値が対象になっているのか
# コイルによる磁束密度は対象になっている
# 行列の式どっかがおかしい

######################################################################
########################################################################


# def F_a_y( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_F")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     F = (math.log(abs(math.sqrt((w-(p_z+c))**2+(v-(p_y+b))**2+(u-(p_x+a))**2)-(w-(p_z+c))))
#         -math.log(abs(math.sqrt((w-(p_z-c))**2+(v-(p_y+b))**2+(u-(p_x+a))**2)-(w-(p_z-c))))
#         -math.log(abs(math.sqrt((w-(p_z+c))**2+(v-(p_y-b))**2+(u-(p_x+a))**2)-(w-(p_z+c))))
#         +math.log(abs(math.sqrt((w-(p_z-c))**2+(v-(p_y-b))**2+(u-(p_x+a))**2)-(w-(p_z-c)))))
#     return F

# def F_a_z( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_F")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     F = (math.log(abs(math.sqrt((v-(p_y+b))**2+(w-(p_z+c))**2+(u-(p_x+a))**2)-(v-(p_y+b))))
#         -math.log(abs(math.sqrt((v-(p_y-b))**2+(w-(p_z+c))**2+(u-(p_x+a))**2)-(v-(p_y-b))))
#         -math.log(abs(math.sqrt((v-(p_y+b))**2+(w-(p_z-c))**2+(u-(p_x+a))**2)-(v-(p_y+b))))
#         +math.log(abs(math.sqrt((v-(p_y-b))**2+(w-(p_z-c))**2+(u-(p_x+a))**2)-(v-(p_y-b)))))
#     return F

# def F_b_y( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_F")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     # print(abs(math.sqrt((w-(p_z+c))**2+(v-(p_y-b))**2+(u-(p_x-a))**2)-(w-(p_z+c))))
#     F = (math.log(abs(math.sqrt((w-(p_z+c))**2+(v-(p_y+b))**2+(u-(p_x-a))**2)-(w-(p_z+c))))
#         -math.log(abs(math.sqrt((w-(p_z-c))**2+(v-(p_y+b))**2+(u-(p_x-a))**2)-(w-(p_z-c))))
#         -math.log(abs(math.sqrt((w-(p_z+c))**2+(v-(p_y-b))**2+(u-(p_x-a))**2)-(w-(p_z+c))))
#         +math.log(abs(math.sqrt((w-(p_z-c))**2+(v-(p_y-b))**2+(u-(p_x-a))**2)-(w-(p_z-c)))))

#     return F

# def F_b_z( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_F")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     F = (math.log(abs(math.sqrt((v-(p_y+b))**2+(w-(p_z+c))**2+(u-(p_x-a))**2)-(v-(p_y+b))))
#         -math.log(abs(math.sqrt((v-(p_y-b))**2+(w-(p_z+c))**2+(u-(p_x-a))**2)-(v-(p_y-b))))
#         -math.log(abs(math.sqrt((v-(p_y+b))**2+(w-(p_z-c))**2+(u-(p_x-a))**2)-(v-(p_y+b))))
#         +math.log(abs(math.sqrt((v-(p_y-b))**2+(w-(p_z-c))**2+(u-(p_x-a))**2)-(v-(p_y-b)))))

#     return F

 
# def F_c_z( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_F")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     F = (math.log(abs(math.sqrt((u-(p_x+a))**2+(w-(p_z+c))**2+(v-(p_y+b))**2)-(u-(p_x+a))))
#         -math.log(abs(math.sqrt((u-(p_x-a))**2+(w-(p_z+c))**2+(v-(p_y+b))**2)-(u-(p_x-a))))
#         -math.log(abs(math.sqrt((u-(p_x+a))**2+(w-(p_z-c))**2+(v-(p_y+b))**2)-(u-(p_x+a))))
#         +math.log(abs(math.sqrt((u-(p_x-a))**2+(w-(p_z-c))**2+(v-(p_y+b))**2)-(u-(p_x-a)))))
#     return F

# def F_c_x( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_F")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     F = (math.log(abs(math.sqrt((w-(p_z+c))**2+(v-(p_y+b))**2+(u-(p_x+a))**2)-(w-(p_z+c))))
#         -math.log(abs(math.sqrt((w-(p_z-c))**2+(v-(p_y+b))**2+(u-(p_x+a))**2)-(w-(p_z-c))))
#         -math.log(abs(math.sqrt((w-(p_z+c))**2+(v-(p_y+b))**2+(u-(p_x-a))**2)-(w-(p_z+c))))
#         +math.log(abs(math.sqrt((w-(p_z-c))**2+(v-(p_y+b))**2+(u-(p_x-a))**2)-(w-(p_z-c)))))
#     return F

# def F_d_z( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_F")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     F = (math.log(abs(math.sqrt((u-(p_x+a))**2+(w-(p_z+c))**2+(v-(p_y-b))**2)-(u-(p_x+a))))
#         -math.log(abs(math.sqrt((u-(p_x-a))**2+(w-(p_z+c))**2+(v-(p_y-b))**2)-(u-(p_x-a))))
#         -math.log(abs(math.sqrt((u-(p_x+a))**2+(w-(p_z-c))**2+(v-(p_y-b))**2)-(u-(p_x+a))))
#         +math.log(abs(math.sqrt((u-(p_x-a))**2+(w-(p_z-c))**2+(v-(p_y-b))**2)-(u-(p_x-a)))))
#     return F



# def F_d_x( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_F")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     F = (math.log(abs(math.sqrt((w-(p_z+c))**2+(v-(p_y-b))**2+(u-(p_x+a))**2)-(w-(p_z+c))))
#         -math.log(abs(math.sqrt((w-(p_z-c))**2+(v-(p_y-b))**2+(u-(p_x+a))**2)-(w-(p_z-c))))
#         -math.log(abs(math.sqrt((w-(p_z+c))**2+(v-(p_y-b))**2+(u-(p_x-a))**2)-(w-(p_z+c))))
#         +math.log(abs(math.sqrt((w-(p_z-c))**2+(v-(p_y-b))**2+(u-(p_x-a))**2)-(w-(p_z-c)))))
#     return F


# def F_e_x( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_F")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     F = (math.log(abs(math.sqrt((v-(p_y+b))**2+(w-(p_z+c))**2+(u-(p_x+a))**2)-(v-(p_y+b))))
#         -math.log(abs(math.sqrt((v-(p_y-b))**2+(w-(p_z+c))**2+(u-(p_x+a))**2)-(v-(p_y-b))))
#         -math.log(abs(math.sqrt((v-(p_y+b))**2+(w-(p_z+c))**2+(u-(p_x-a))**2)-(v-(p_y+b))))
#         +math.log(abs(math.sqrt((v-(p_y-b))**2+(w-(p_z+c))**2+(u-(p_x-a))**2)-(v-(p_y-b)))))
#     return F


# def F_e_y( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_F")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     F = (math.log(abs(math.sqrt((u-(p_x+a))**2+(w-(p_z+c))**2+(v-(p_y+b))**2)-(u-(p_x+a))))
#         -math.log(abs(math.sqrt((u-(p_x-a))**2+(w-(p_z+c))**2+(v-(p_y+b))**2)-(u-(p_x-a))))
#         -math.log(abs(math.sqrt((u-(p_x+a))**2+(w-(p_z+c))**2+(v-(p_y-b))**2)-(u-(p_x+a))))
#         +math.log(abs(math.sqrt((u-(p_x-a))**2+(w-(p_z+c))**2+(v-(p_y-b))**2)-(u-(p_x-a)))))
#     return F


# def F_f_x( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_F")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     F = (math.log(abs(math.sqrt((v-(p_y+b))**2+(w-(p_z-c))**2+(u-(p_x+a))**2)-(v-(p_y+b))))
#         -math.log(abs(math.sqrt((v-(p_y-b))**2+(w-(p_z-c))**2+(u-(p_x+a))**2)-(v-(p_y-b))))
#         -math.log(abs(math.sqrt((v-(p_y+b))**2+(w-(p_z-c))**2+(u-(p_x-a))**2)-(v-(p_y+b))))
#         +math.log(abs(math.sqrt((v-(p_y-b))**2+(w-(p_z-c))**2+(u-(p_x-a))**2)-(v-(p_y-b)))))
#     return F


# def F_f_y( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_F")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     F = (math.log(abs(math.sqrt((u-(p_x+a))**2+(w-(p_z-c))**2+(v-(p_y+b))**2)-(u-(p_x+a))))
#         -math.log(abs(math.sqrt((u-(p_x-a))**2+(w-(p_z-c))**2+(v-(p_y+b))**2)-(u-(p_x-a))))
#         -math.log(abs(math.sqrt((u-(p_x+a))**2+(w-(p_z-c))**2+(v-(p_y-b))**2)-(u-(p_x+a))))
#         +math.log(abs(math.sqrt((u-(p_x-a))**2+(w-(p_z-c))**2+(v-(p_y-b))**2)-(u-(p_x-a)))))
#     return F

def F_a_y( u, v, w, a, b, c, p : np.ndarray):
    # print("function_F")
    p_x, p_y, p_z = p[0], p[1], p[2]
    r_1p = ((u - (p_x + a))**2 + (v - (p_y + b))**2 + (w - (p_z - c))**2)**0.5
    r_2p = ((u - (p_x + a))**2 + (v - (p_y + b))**2 + (w - (p_z + c))**2)**0.5
    r_3p = ((u - (p_x + a))**2 + (v - (p_y - b))**2 + (w - (p_z + c))**2)**0.5
    r_4p = ((u - (p_x + a))**2 + (v - (p_y - b))**2 + (w - (p_z - c))**2)**0.5
    y_1p = v - (p_y + b)
    y_2p = v - (p_y + b)
    y_3p = v - (p_y - b)
    y_4p = v - (p_y - b)
    d1 = r_1p - y_1p
    d2 = r_2p - y_2p
    d3 = r_3p - y_3p
    d4 = r_4p - y_4p
    if d1 <= 0:
        d1 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d2 <= 0:
        d2 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d3 <= 0:
        d3 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d4 <= 0:
        d4 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    F = - math.log(d1) + math.log(d2) - math.log(d3) + math.log(d4)
    return F

def F_a_z( u, v, w, a, b, c, p : np.ndarray):
    # print("function_F")
    p_x, p_y, p_z = p[0], p[1], p[2]
    r_1p = ((u - (p_x + a))**2 + (v - (p_y + b))**2 + (w - (p_z - c))**2)**0.5
    r_2p = ((u - (p_x + a))**2 + (v - (p_y + b))**2 + (w - (p_z + c))**2)**0.5
    r_3p = ((u - (p_x + a))**2 + (v - (p_y - b))**2 + (w - (p_z + c))**2)**0.5
    r_4p = ((u - (p_x + a))**2 + (v - (p_y - b))**2 + (w - (p_z - c))**2)**0.5
    z_1p = w - (p_z - c)
    z_2p = w - (p_z + c)
    z_3p = w - (p_z + c)
    z_4p = w - (p_z - c)
    d1 = r_1p - z_1p
    d2 = r_2p - z_2p
    d3 = r_3p - z_3p
    d4 = r_4p - z_4p
    # print(d1)
    # print(d2)
    # print(d3)
    # print(d4)
    if d1 <= 0:
        d1 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d2 <= 0:
        d2 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d3 <= 0:
        d3 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d4 <= 0:
        d4 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    F = -math.log(d1) + math.log(d2) - math.log(d3) + math.log(d4)
    return F

def F_b_y( u, v, w, a, b, c, p : np.ndarray):
    # print("function_F")
    p_x, p_y, p_z = p[0], p[1], p[2]
    r_1p = ((u - (p_x - a))**2 + (v - (p_y + b))**2 + (w - (p_z - c))**2)**0.5
    r_2p = ((u - (p_x - a))**2 + (v - (p_y + b))**2 + (w - (p_z + c))**2)**0.5
    r_3p = ((u - (p_x - a))**2 + (v - (p_y - b))**2 + (w - (p_z + c))**2)**0.5
    r_4p = ((u - (p_x - a))**2 + (v - (p_y - b))**2 + (w - (p_z - c))**2)**0.5
    y_1p = v - (p_y + b)
    y_2p = v - (p_y + b)
    y_3p = v - (p_y - b)
    y_4p = v - (p_y - b)
    d1 = r_1p - y_1p
    d2 = r_2p - y_2p
    d3 = r_3p - y_3p
    d4 = r_4p - y_4p
    if d1 <= 0:
        d1 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d2 <= 0:
        d2 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d3 <= 0:
        d3 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d4 <= 0:
        d4 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    F = -math.log(d1) + math.log(d2) - math.log(d3) + math.log(d4)
    return F

def F_b_z( u, v, w, a, b, c, p : np.ndarray):
    # print("function_F")
    p_x, p_y, p_z = p[0], p[1], p[2]
    r_1p = ((u - (p_x - a))**2 + (v - (p_y + b))**2 + (w - (p_z - c))**2)**0.5
    r_2p = ((u - (p_x - a))**2 + (v - (p_y + b))**2 + (w - (p_z + c))**2)**0.5
    r_3p = ((u - (p_x - a))**2 + (v - (p_y - b))**2 + (w - (p_z + c))**2)**0.5
    r_4p = ((u - (p_x - a))**2 + (v - (p_y - b))**2 + (w - (p_z - c))**2)**0.5
    z_1p = w - (p_z - c)
    z_2p = w - (p_z + c)
    z_3p = w - (p_z + c)
    z_4p = w - (p_z - c)
    d1 = r_1p - z_1p
    d2 = r_2p - z_2p
    d3 = r_3p - z_3p
    d4 = r_4p - z_4p
    # print(d1)
    # print(d2)
    # print(d3)
    # print(d4)
    if d1 <= 0:
        d1 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d2 <= 0:
        d2 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d3 <= 0:
        d3 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d4 <= 0:
        d4 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    F = -math.log(d1) + math.log(d2) - math.log(d3) + math.log(d4)
    return F

def F_c_z( u, v, w, a, b, c, p : np.ndarray):
    # print("function_F")
    p_x, p_y, p_z = p[0], p[1], p[2]
    r_1p = ((u - (p_x - a))**2 + (v - (p_y + b))**2 + (w - (p_z + c))**2)**0.5
    r_2p = ((u - (p_x + a))**2 + (v - (p_y + b))**2 + (w - (p_z + c))**2)**0.5
    r_3p = ((u - (p_x + a))**2 + (v - (p_y + b))**2 + (w - (p_z - c))**2)**0.5
    r_4p = ((u - (p_x - a))**2 + (v - (p_y + b))**2 + (w - (p_z - c))**2)**0.5
    z_1p = w - (p_z + c)
    z_2p = w - (p_z + c)
    z_3p = w - (p_z - c)
    z_4p = w - (p_z - c)
    d1 = r_1p - z_1p
    d2 = r_2p - z_2p
    d3 = r_3p - z_3p
    d4 = r_4p - z_4p
    if d1 <= 0:
        d1 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d2 <= 0:
        d2 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d3 <= 0:
        d3 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d4 <= 0:
        d4 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    F = -math.log(d1) + math.log(d2) - math.log(d3) + math.log(d4)
    return F

def F_c_x( u, v, w, a, b, c, p : np.ndarray):
    # print("function_F")
    p_x, p_y, p_z = p[0], p[1], p[2]
    r_1p = ((u - (p_x - a))**2 + (v - (p_y + b))**2 + (w - (p_z + c))**2)**0.5
    r_2p = ((u - (p_x + a))**2 + (v - (p_y + b))**2 + (w - (p_z + c))**2)**0.5
    r_3p = ((u - (p_x + a))**2 + (v - (p_y + b))**2 + (w - (p_z - c))**2)**0.5
    r_4p = ((u - (p_x - a))**2 + (v - (p_y + b))**2 + (w - (p_z - c))**2)**0.5
    x_1p = u - (p_x - a)
    x_2p = u - (p_x + a)
    x_3p = u - (p_x + a)
    x_4p = u - (p_x - a)
    d1 = r_1p - x_1p
    d2 = r_2p - x_2p
    d3 = r_3p - x_3p
    d4 = r_4p - x_4p
    if d1 <= 0:
        d1 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d2 <= 0:
        d2 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d3 <= 0:
        d3 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d4 <= 0:
        d4 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    F = -math.log(d1) + math.log(d2) - math.log(d3) + math.log(d4)
    return F

def F_d_z( u, v, w, a, b, c, p : np.ndarray):
    # print("function_F")
    p_x, p_y, p_z = p[0], p[1], p[2]
    r_1p = ((u - (p_x - a))**2 + (v - (p_y - b))**2 + (w - (p_z + c))**2)**0.5
    r_2p = ((u - (p_x + a))**2 + (v - (p_y - b))**2 + (w - (p_z + c))**2)**0.5
    r_3p = ((u - (p_x + a))**2 + (v - (p_y - b))**2 + (w - (p_z - c))**2)**0.5
    r_4p = ((u - (p_x - a))**2 + (v - (p_y - b))**2 + (w - (p_z - c))**2)**0.5
    z_1p = w - (p_z + c)
    z_2p = w - (p_z + c)
    z_3p = w - (p_z - c)
    z_4p = w - (p_z - c)
    d1 = r_1p - z_1p
    d2 = r_2p - z_2p
    d3 = r_3p - z_3p
    d4 = r_4p - z_4p
    if d1 <= 0:
        d1 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d2 <= 0:
        d2 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d3 <= 0:
        d3 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d4 <= 0:
        d4 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    F = - math.log(d1) + math.log(d2) - math.log(d3) + math.log(d4)
    return F



def F_d_x( u, v, w, a, b, c, p : np.ndarray):
    # print("function_F")
    p_x, p_y, p_z = p[0], p[1], p[2]
    r_1p = ((u - (p_x - a))**2 + (v - (p_y - b))**2 + (w - (p_z + c))**2)**0.5
    r_2p = ((u - (p_x + a))**2 + (v - (p_y - b))**2 + (w - (p_z + c))**2)**0.5
    r_3p = ((u - (p_x + a))**2 + (v - (p_y - b))**2 + (w - (p_z - c))**2)**0.5
    r_4p = ((u - (p_x - a))**2 + (v - (p_y - b))**2 + (w - (p_z - c))**2)**0.5
    x_1p = u - (p_x - a)
    x_2p = u - (p_x + a)
    x_3p = u - (p_x + a)
    x_4p = u - (p_x - a)
    d1 = r_1p - x_1p
    d2 = r_2p - x_2p
    d3 = r_3p - x_3p
    d4 = r_4p - x_4p
    if d1 <= 0:
        d1 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d2 <= 0:
        d2 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d3 <= 0:
        d3 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d4 <= 0:
        d4 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    F = -math.log(d1) + math.log(d2) - math.log(d3) + math.log(d4)
    return F


def F_e_x( u, v, w, a, b, c, p : np.ndarray):
    # print("function_F")
    p_x, p_y, p_z = p[0], p[1], p[2]
    r_1p = ((u - (p_x + a))**2 + (v - (p_y - b))**2 + (w - (p_z + c))**2)**0.5
    r_2p = ((u - (p_x + a))**2 + (v - (p_y + b))**2 + (w - (p_z + c))**2)**0.5
    r_3p = ((u - (p_x - a))**2 + (v - (p_y + b))**2 + (w - (p_z + c))**2)**0.5
    r_4p = ((u - (p_x - a))**2 + (v - (p_y - b))**2 + (w - (p_z + c))**2)**0.5
    x_1p = u - (p_x + a)
    x_2p = u - (p_x + a)
    x_3p = u - (p_x - a)
    x_4p = u - (p_x - a)
    d1 = r_1p - x_1p
    d2 = r_2p - x_2p
    d3 = r_3p - x_3p
    d4 = r_4p - x_4p
    if d1 <= 0:
        d1 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d2 <= 0:
        d2 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d3 <= 0:
        d3 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d4 <= 0:
        d4 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    F = -math.log(d1) + math.log(d2) - math.log(d3) + math.log(d4)
    return F


def F_e_y( u, v, w, a, b, c, p : np.ndarray):
    # print("function_F")
    p_x, p_y, p_z = p[0], p[1], p[2]
    r_1p = ((u - (p_x + a))**2 + (v - (p_y - b))**2 + (w - (p_z + c))**2)**0.5
    r_2p = ((u - (p_x + a))**2 + (v - (p_y + b))**2 + (w - (p_z + c))**2)**0.5
    r_3p = ((u - (p_x - a))**2 + (v - (p_y + b))**2 + (w - (p_z + c))**2)**0.5
    r_4p = ((u - (p_x - a))**2 + (v - (p_y - b))**2 + (w - (p_z + c))**2)**0.5
    y_1p = v - (p_y - b)
    y_2p = v - (p_y + b)
    y_3p = v - (p_y + b)
    y_4p = v - (p_y - b)
    d1 = r_1p - y_1p
    d2 = r_2p - y_2p
    d3 = r_3p - y_3p
    d4 = r_4p - y_4p
    if d1 <= 0:
        d1 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d2 <= 0:
        d2 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d3 <= 0:
        d3 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d4 <= 0:
        d4 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    F = -math.log(d1) + math.log(d2) - math.log(d3) + math.log(d4)
    return F


def F_f_x( u, v, w, a, b, c, p : np.ndarray):
    # print("function_F")
    p_x, p_y, p_z = p[0], p[1], p[2]
    r_1p = ((u - (p_x + a))**2 + (v - (p_y - b))**2 + (w - (p_z - c))**2)**0.5
    r_2p = ((u - (p_x + a))**2 + (v - (p_y + b))**2 + (w - (p_z - c))**2)**0.5
    r_3p = ((u - (p_x - a))**2 + (v - (p_y + b))**2 + (w - (p_z - c))**2)**0.5
    r_4p = ((u - (p_x - a))**2 + (v - (p_y - b))**2 + (w - (p_z - c))**2)**0.5
    x_1p = u - (p_x + a)
    x_2p = u - (p_x + a)
    x_3p = u - (p_x - a)
    x_4p = u - (p_x - a)
    d1 = r_1p - x_1p
    d2 = r_2p - x_2p
    d3 = r_3p - x_3p
    d4 = r_4p - x_4p
    if d1 <= 0:
        d1 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d2 <= 0:
        d2 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d3 <= 0:
        d3 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d4 <= 0:
        d4 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    F = -math.log(d1) + math.log(d2) - math.log(d3) + math.log(d4)
    return F


def F_f_y( u, v, w, a, b, c, p : np.ndarray):
    # print("function_F")
    p_x, p_y, p_z = p[0], p[1], p[2]
    r_1p = ((u - (p_x + a))**2 + (v - (p_y - b))**2 + (w - (p_z - c))**2)**0.5
    r_2p = ((u - (p_x + a))**2 + (v - (p_y + b))**2 + (w - (p_z - c))**2)**0.5
    r_3p = ((u - (p_x - a))**2 + (v - (p_y + b))**2 + (w - (p_z - c))**2)**0.5
    r_4p = ((u - (p_x - a))**2 + (v - (p_y - b))**2 + (w - (p_z - c))**2)**0.5
    y_1p = v - (p_y - b)
    y_2p = v - (p_y + b)
    y_3p = v - (p_y + b)
    y_4p = v - (p_y - b)
    d1 = r_1p - y_1p
    d2 = r_2p - y_2p
    d3 = r_3p - y_3p
    d4 = r_4p - y_4p
    if d1 <= 0:
        d1 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d2 <= 0:
        d2 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d3 <= 0:
        d3 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    if d4 <= 0:
        d4 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    F = -math.log(d1) + math.log(d2) - math.log(d3) + math.log(d4)
    return F


#########################
# [0,2π]でarctanを計算する関数
def atan_2pi( y, x):
    rad = math.atan2(y, x)
    if rad < 0:
        rad = 2 * np.pi + rad
    return rad

#####################################
# G_~_ の範囲条件の変更
# arctan の定義域の変更←やっぱ[-pi,pi]っぽい　じゃないと小要素が自分自身に影響与える
# G　に　Σ追加


##############################################

def G_a( u, v, w, a, b, c, p : np.ndarray, flag :bool):
    # print("function_G")
    p_x, p_y, p_z = p[0], p[1], p[2]
    s1,s2,s3 = 1,1,1
    if u-p_x-a < 0:
        s1 = -1
    if v-p_y-b < 0:
        s2 = -1
    if v-p_y+b < 0:
        s3 = -1
    if flag is True:
        print("a s1",s1,"s2",s2,"s3",s3)
    G = ((-1*s1*s2*math.atan2( abs(v-(p_y+b))*(w-(p_z+c)) , abs(u-(p_x+a))*math.sqrt((w-(p_z+c))**2+(v-(p_y+b))**2+(u-(p_x+a))**2)))
        -(-1*s1*s2*math.atan2( abs(v-(p_y+b))*(w-(p_z-c)) , abs(u-(p_x+a))*math.sqrt((w-(p_z-c))**2+(v-(p_y+b))**2+(u-(p_x+a))**2)))
        +(s1*s3*math.atan2( abs(v-(p_y-b))*(w-(p_z+c)) , abs(u-(p_x+a))*math.sqrt((w-(p_z+c))**2+(v-(p_y-b))**2+(u-(p_x+a))**2)))
        -(s1*s3*math.atan2( abs(v-(p_y-b))*(w-(p_z-c)) , abs(u-(p_x+a))*math.sqrt((w-(p_z-c))**2+(v-(p_y-b))**2+(u-(p_x+a))**2))))
    # print(abs(v-(p_y+b))*(w-(p_z+c)))
    # print(abs(u-(p_x+a))*math.sqrt((w-(p_z+c))**2+(v-(p_y+b))**2+(u-(p_x+a))**2))
    # print((-1*s1*s2*math.atan2( abs(v-(p_y+b))*(w-(p_z+c)) , abs(u-(p_x+a))*math.sqrt((w-(p_z+c))**2+(v-(p_y+b))**2+(u-(p_x+a))**2))))
    # print(-(-1*s1*s2*math.atan2( abs(v-(p_y+b))*(w-(p_z-c)) , abs(u-(p_x+a))*math.sqrt((w-(p_z-c))**2+(v-(p_y+b))**2+(u-(p_x+a))**2))))
    # print((s1*s3*math.atan2( abs(v-(p_y-b))*(w-(p_z+c)) , abs(u-(p_x+a))*math.sqrt((w-(p_z+c))**2+(v-(p_y-b))**2+(u-(p_x+a))**2))))
    # print(-(s1*s3*math.atan2( abs(v-(p_y-b))*(w-(p_z-c)) , abs(u-(p_x+a))*math.sqrt((w-(p_z-c))**2+(v-(p_y-b))**2+(u-(p_x+a))**2))))
    return G

def G_b( u, v, w, a, b, c, p : np.ndarray, flag :bool):
    # print("function_G")
    p_x, p_y, p_z = p[0], p[1], p[2]
    s1,s2,s3 = 1,1,1
    if u-p_x+a < 0:
        s1 = -1
    if v-p_y-b < 0:
        s2 = -1
    if v-p_y+b < 0:
        s3 = -1
    if flag is True:
        print("b s1",s1,"s2",s2,"s3",s3)
    G = ((-1*s1*s2*math.atan2( abs(v-(p_y+b))*(w-(p_z+c)), abs(u-(p_x-a))*math.sqrt((w-(p_z+c))**2+(v-(p_y+b))**2+(u-(p_x-a))**2)))
        -(-1*s1*s2*math.atan2( abs(v-(p_y+b))*(w-(p_z-c)), abs(u-(p_x-a))*math.sqrt((w-(p_z-c))**2+(v-(p_y+b))**2+(u-(p_x-a))**2)))
        +(s1*s3*math.atan2( abs(v-(p_y-b))*(w-(p_z+c)), abs(u-(p_x-a))*math.sqrt((w-(p_z+c))**2+(v-(p_y-b))**2+(u-(p_x-a))**2)))
        -(s1*s3*math.atan2( abs(v-(p_y-b))*(w-(p_z-c)), abs(u-(p_x-a))*math.sqrt((w-(p_z-c))**2+(v-(p_y-b))**2+(u-(p_x-a))**2))))
    return G

def G_c( u, v, w, a, b, c, p : np.ndarray, flag :bool):
    # print("function_G")
    p_x, p_y, p_z = p[0], p[1], p[2]
    s1,s2,s3 = 1,1,1
    if v-(p_y+b) < 0:
        s1 = -1
    if w-p_z-c < 0:
        s2 = -1
    if w-p_z+c < 0:
        s3 = -1
    if flag is True:
        print("c s1",s1,"s2",s2,"s3",s3)
    G = ((-1*s1*s2*math.atan2( abs(w-p_z-c)*(u-(p_x+a)), abs(v-(p_y+b))*math.sqrt((u-(p_x+a))**2+(w-p_z-c)**2+(v-(p_y+b))**2)))
        -(-1*s1*s2*math.atan2( abs(w-p_z-c)*(u-(p_x-a)), abs(v-(p_y+b))*math.sqrt((u-(p_x-a))**2+(w-p_z-c)**2+(v-(p_y+b))**2)))
        +(s1*s3*math.atan2( abs(w-p_z+c)*(u-(p_x+a)), abs(v-(p_y+b))*math.sqrt((u-(p_x+a))**2+(w-p_z+c)**2+(v-(p_y+b))**2)))
        -(s1*s3*math.atan2( abs(w-p_z+c)*(u-(p_x-a)), abs(v-(p_y+b))*math.sqrt((u-(p_x-a))**2+(w-p_z+c)**2+(v-(p_y+b))**2))))
    return G

def G_d( u, v, w, a, b, c, p : np.ndarray, flag :bool):
    # print("function_G")
    p_x, p_y, p_z = p[0], p[1], p[2]
    s1,s2,s3 = 1,1,1
    if v-(p_y-b) < 0:
        s1 = -1
    if w-p_z-c < 0:
        s2 = -1
    if w-p_z+c < 0:
        s3 = -1
    if flag is True:
        print("d s1",s1,"s2",s2,"s3",s3)
    G = ((-1*s1*s2*math.atan2( abs(w-p_z-c)*(u-(p_x+a)), abs(v-(p_y-b))*math.sqrt((u-(p_x+a))**2+(w-p_z-c)**2+(v-(p_y-b))**2)))
        -(-1*s1*s2*math.atan2( abs(w-p_z-c)*(u-(p_x-a)), abs(v-(p_y-b))*math.sqrt((u-(p_x-a))**2+(w-p_z-c)**2+(v-(p_y-b))**2)))
        +(s1*s3*math.atan2( abs(w-p_z+c)*(u-(p_x+a)), abs(v-(p_y-b))*math.sqrt((u-(p_x+a))**2+(w-p_z+c)**2+(v-(p_y-b))**2)))
        -(s1*s3*math.atan2( abs(w-p_z+c)*(u-(p_x-a)), abs(v-(p_y-b))*math.sqrt((u-(p_x-a))**2+(w-p_z+c)**2+(v-(p_y-b))**2))))
    return G

def G_e( u, v, w, a, b, c, p : np.ndarray, flag :bool):
    # print("function_G")
    p_x, p_y, p_z = p[0], p[1], p[2]
    s1,s2,s3 = 1,1,1
    if w-p_z-c < 0:
        s1 = -1
    if u-p_x-a < 0:
        s2 = -1
    if u-p_x+a < 0:
        s3 = -1
    if flag is True:
        print("e s1",s1,"s2",s2,"s3",s3)
    G = ((-1*s1*s2*math.atan2( abs(u-p_x-a)*(v-(p_y+b)), abs(w-p_z-c)*math.sqrt((v-(p_y+b))**2+(w-p_z-c)**2+(u-p_x-a)**2)))
        -(-1*s1*s2*math.atan2( abs(u-p_x-a)*(v-(p_y-b)), abs(w-p_z-c)*math.sqrt((v-(p_y-b))**2+(w-p_z-c)**2+(u-p_x-a)**2)))
        +(s1*s3*math.atan2( abs(u-p_x+a)*(v-(p_y+b)),  abs(w-p_z-c)*math.sqrt((v-(p_y+b))**2+(w-p_z-c)**2+(u-p_x+a)**2)))
        -(s1*s3*math.atan2( abs(u-p_x+a)*(v-(p_y-b)),  abs(w-p_z-c)*math.sqrt((v-(p_y-b))**2+(w-p_z-c)**2+(u-p_x+a)**2))))
    return G

def G_f( u, v, w, a, b, c, p : np.ndarray, flag :bool):
    # print("function_G")
    p_x, p_y, p_z = p[0], p[1], p[2]
    s1,s2,s3 = 1,1,1
    if w-p_z+c < 0:
        s1 = -1
    if u-p_x-a < 0:
        s2 = -1
    if u-p_x+a < 0:
        s3 = -1
    if flag is True:
        print("f s1",s1,"s2",s2,"s3",s3)
    G = ((-1*s1*s2*math.atan2( abs(u-p_x-a)*(v-(p_y+b)), abs(w-p_z+c)*math.sqrt((v-(p_y+b))**2+(w-p_z+c)**2+(u-p_x-a)**2)))
        -(-1*s1*s2*math.atan2( abs(u-p_x-a)*(v-(p_y-b)), abs(w-p_z+c)*math.sqrt((v-(p_y-b))**2+(w-p_z+c)**2+(u-p_x-a)**2)))
        +(s1*s3*math.atan2( abs(u-p_x+a)*(v-(p_y+b)), abs(w-p_z+c)*math.sqrt((v-(p_y+b))**2+(w-p_z+c)**2+(u-p_x+a)**2)))
        -(s1*s3*math.atan2( abs(u-p_x+a)*(v-(p_y-b)), abs(w-p_z+c)*math.sqrt((v-(p_y-b))**2+(w-p_z+c)**2+(u-p_x+a)**2))))
    return G

#######################################################

def Sign(x):
    s = 0
    if x >= 0:
        s = 1
    else:
        s = -1
    return s

###############
# def G_a( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_G")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     G = (((-u+(p_x+a))*(v-(p_y+b))*math.atan2( abs(v-(p_y+b))*(w-(p_z+c)) , abs(u-(p_x+a))*math.sqrt((w-(p_z+c))**2+(v-(p_y+b))**2+(u-(p_x+a))**2)))/(abs(u-(p_x+a))*abs(v-(p_y+b)))
#         -((-u+(p_x+a))*(v-(p_y+b))*math.atan2( abs(v-(p_y+b))*(w-(p_z-c)) , abs(u-(p_x+a))*math.sqrt((w-(p_z-c))**2+(v-(p_y+b))**2+(u-(p_x+a))**2)))/(abs(u-(p_x+a))*abs(v-(p_y+b)))
#         +((u-(p_x+a))*(v-(p_y-b))*math.atan2( abs(v-(p_y-b))*(w-(p_z+c)) , abs(u-(p_x+a))*math.sqrt((w-(p_z+c))**2+(v-(p_y-b))**2+(u-(p_x+a))**2)))/(abs(u-(p_x+a))*abs(v-(p_y-b)))
#         -((u-(p_x+a))*(v-(p_y-b))*math.atan2( abs(v-(p_y-b))*(w-(p_z-c)) , abs(u-(p_x+a))*math.sqrt((w-(p_z-c))**2+(v-(p_y-b))**2+(u-(p_x+a))**2)))/(abs(u-(p_x+a))*abs(v-(p_y-b))))
#     return G

# def G_b( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_G")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     G = (((-u+(p_x-a))*(v-(p_y+b))*math.atan2( abs(v-(p_y+b))*(w-(p_z+c)), abs(u-(p_x-a))*math.sqrt((w-(p_z+c))**2+(v-(p_y+b))**2+(u-(p_x-a))**2)))/(abs(u-(p_x-a))*abs(v-(p_y+b)))
#         -((-u+(p_x-a))*(v-(p_y+b))*math.atan2( abs(v-(p_y+b))*(w-(p_z-c)), abs(u-(p_x-a))*math.sqrt((w-(p_z-c))**2+(v-(p_y+b))**2+(u-(p_x-a))**2)))/(abs(u-(p_x-a))*abs(v-(p_y+b)))
#         +((u-(p_x-a))*(v-(p_y-b))*math.atan2( abs(v-(p_y-b))*(w-(p_z+c)), abs(u-(p_x-a))*math.sqrt((w-(p_z+c))**2+(v-(p_y-b))**2+(u-(p_x-a))**2)))/(abs(u-(p_x-a))*abs(v-(p_y-b)))
#         -((u-(p_x-a))*(v-(p_y-b))*math.atan2( abs(v-(p_y-b))*(w-(p_z-c)), abs(u-(p_x-a))*math.sqrt((w-(p_z-c))**2+(v-(p_y-b))**2+(u-(p_x-a))**2)))/(abs(u-(p_x-a))*abs(v-(p_y-b))))
#     return G

# def G_c( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_G")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     G = (((-v+(p_y+b))*(w-p_z-c)*math.atan2( abs(w-p_z-c)*(u-(p_x+a)), abs(v-(p_y+b))*math.sqrt((u-(p_x+a))**2+(w-p_z-c)**2+(v-(p_y+b))**2)))/(abs(v-(p_y+b))*abs(w-p_z-c))
#         -((-v+(p_y+b))*(w-p_z-c)*math.atan2( abs(w-p_z-c)*(u-(p_x-a)), abs(v-(p_y+b))*math.sqrt((u-(p_x-a))**2+(w-p_z-c)**2+(v-(p_y+b))**2)))/(abs(v-(p_y+b))*abs(w-p_z-c))
#         +((v-(p_y+b))*(w-p_z+c)*math.atan2( abs(w-p_z+c)*(u-(p_x+a)), abs(v-(p_y+b))*math.sqrt((u-(p_x+a))**2+(w-p_z+c)**2+(v-(p_y+b))**2)))/(abs(v-(p_y+b))*abs(w-p_z+c))
#         -((v-(p_y+b))*(w-p_z+c)*math.atan2( abs(w-p_z+c)*(u-(p_x-a)), abs(v-(p_y+b))*math.sqrt((u-(p_x-a))**2+(w-p_z+c)**2+(v-(p_y+b))**2)))/(abs(v-(p_y+b))*abs(w-p_z+c)))
#     return G

# def G_d( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_G")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     G = (((-v+(p_y-b))*(w-p_z-c)*math.atan2( abs(w-p_z-c)*(u-(p_x+a)), abs(v-(p_y-b))*math.sqrt((u-(p_x+a))**2+(w-p_z-c)**2+(v-(p_y-b))**2)))/(abs(v-(p_y-b))*abs(w-p_z-c))
#         -((-v+(p_y-b))*(w-p_z-c)*math.atan2( abs(w-p_z-c)*(u-(p_x-a)), abs(v-(p_y-b))*math.sqrt((u-(p_x-a))**2+(w-p_z-c)**2+(v-(p_y-b))**2)))/(abs(v-(p_y-b))*abs(w-p_z-c))
#         +((v-(p_y-b))*(w-p_z+c)*math.atan2( abs(w-p_z+c)*(u-(p_x+a)), abs(v-(p_y-b))*math.sqrt((u-(p_x+a))**2+(w-p_z+c)**2+(v-(p_y-b))**2)))/(abs(v-(p_y-b))*abs(w-p_z+c))
#         -((v-(p_y-b))*(w-p_z+c)*math.atan2( abs(w-p_z+c)*(u-(p_x-a)), abs(v-(p_y-b))*math.sqrt((u-(p_x-a))**2+(w-p_z+c)**2+(v-(p_y-b))**2)))/(abs(v-(p_y-b))*abs(w-p_z+c)))
#     return G

# def G_e( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_G")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     G = (((-u+p_x+a)*(w-(p_z+c))*math.atan2( abs(u-p_x-a)*(v-(p_y+b)), abs(w-p_z-c)*math.sqrt((v-(p_y+b))**2+(w-p_z-c)**2+(u-p_x-a)**2)))/(abs(u-p_x-a)*abs(w-p_z-c))
#         -((-u+p_x+a)*(w-(p_z+c))*math.atan2( abs(u-p_x-a)*(v-(p_y-b)), abs(w-p_z-c)*math.sqrt((v-(p_y-b))**2+(w-p_z-c)**2+(u-p_x-a)**2)))/(abs(u-p_x-a)*abs(w-p_z-c))
#         +((u-p_x+a)*(w-(p_z+c))*math.atan2( abs(u-p_x+a)*(v-(p_y+b)),  abs(w-p_z-c)*math.sqrt((v-(p_y+b))**2+(w-p_z-c)**2+(u-p_x+a)**2)))/(abs(u-p_x+a)*abs(w-p_z-c))
#         -((u-p_x+a)*(w-(p_z+c))*math.atan2( abs(u-p_x+a)*(v-(p_y-b)),  abs(w-p_z-c)*math.sqrt((v-(p_y-b))**2+(w-p_z-c)**2+(u-p_x+a)**2)))/(abs(u-p_x+a)*abs(w-p_z-c)))
#     return G

# def G_f( u, v, w, a, b, c, p : np.ndarray):
#     # print("function_G")
#     p_x, p_y, p_z = p[0], p[1], p[2]
#     G = (((-u+p_x+a)*(w-(p_z-c))*math.atan2( abs(u-p_x-a)*(v-(p_y+b)), abs(w-p_z+c)*math.sqrt((v-(p_y+b))**2+(w-p_z+c)**2+(u-p_x-a)**2)))/(abs(u-p_x-a)*abs(w-p_z+c))
#         -((-u+p_x+a)*(w-(p_z-c))*math.atan2( abs(u-p_x-a)*(v-(p_y-b)), abs(w-p_z+c)*math.sqrt((v-(p_y-b))**2+(w-p_z+c)**2+(u-p_x-a)**2)))/(abs(u-p_x-a)*abs(w-p_z+c))
#         +((u-p_x+a)*(w-(p_z-c))*math.atan2( abs(u-p_x+a)*(v-(p_y+b)), abs(w-p_z+c)*math.sqrt((v-(p_y+b))**2+(w-p_z+c)**2+(u-p_x+a)**2)))/(abs(u-p_x+a)*abs(w-p_z+c))
#         -((u-p_x+a)*(w-(p_z-c))*math.atan2( abs(u-p_x+a)*(v-(p_y-b)), abs(w-p_z+c)*math.sqrt((v-(p_y-b))**2+(w-p_z+c)**2+(u-p_x+a)**2)))/(abs(u-p_x+a)*abs(w-p_z+c)))
#     return G



######################################################

# j_cen : iに磁界を起こす要素jの中心(1×3)
# i_cen : 磁界を求める要素の中心(1×3)
# alph_~ : jによりiに発生する磁界の~(x,y,z)成分のjx(jの磁化ベクトルのx成分)の係数
# beta_~ : jによりiに発生する磁界の~(x,y,z)成分のjy(jの磁化ベクトルのy成分)の係数
# gam_~ : jによりiに発生する磁界の~(x,y,z)成分のjz(jの磁化ベクトルのz成分)の係数

#####################################################
# 係数が本当にこれであっているか確認

def function_alpha_x(j_cen : np.ndarray, a, b, c, i_cen : np.ndarray ,flag:bool):
    u,v,w = i_cen[0], i_cen[1], i_cen[2]
    # print(G_c( u, v, w, a, b, c, j_cen))
    # print(G_d( u, v, w, a, b, c, j_cen))
    # print( G_e( u, v, w, a, b, c, j_cen))
    # print(G_f( u, v, w, a, b, c, j_cen))
    alph_x = G_c( u, v, w, a, b, c, j_cen, flag) - G_d( u, v, w, a, b, c, j_cen, flag) + G_e( u, v, w, a, b, c, j_cen, flag) - G_f( u, v, w, a, b, c, j_cen, flag)
    return alph_x

def function_beta_x(j_cen : np.ndarray, a, b, c, i_cen : np.ndarray):
    u,v,w = i_cen[0], i_cen[1], i_cen[2]
    beta_x = F_a_z(u, v, w, a, b, c, j_cen) - F_b_z(u, v, w, a, b, c, j_cen)
    return beta_x

def function_gamma_x(j_cen : np.ndarray, a, b, c, i_cen : np.ndarray):
    u,v,w = i_cen[0], i_cen[1], i_cen[2]
    gam_x = F_a_y(u, v, w, a, b, c, j_cen) - F_b_y(u, v, w, a, b, c, j_cen)
    return gam_x

def function_alpha_y(j_cen : np.ndarray, a, b, c, i_cen : np.ndarray):
    u,v,w = i_cen[0], i_cen[1], i_cen[2]
    alph_y = F_c_z(u, v, w, a, b, c, j_cen) - F_d_z(u, v, w, a, b, c, j_cen)
    return alph_y

def function_beta_y(j_cen : np.ndarray, a, b, c, i_cen : np.ndarray, flag):
    u,v,w = i_cen[0], i_cen[1], i_cen[2]
    beta_y = G_a( u, v, w ,a, b, c, j_cen, flag) - G_b( u, v, w ,a, b, c, j_cen, flag) + G_e(u, v, w, a, b, c, j_cen, flag) - G_f(u, v, w, a, b, c, j_cen, flag)
    return beta_y

def function_gamma_y(j_cen : np.ndarray, a, b, c, i_cen : np.ndarray):
    u,v,w = i_cen[0], i_cen[1], i_cen[2]
    gam_y = F_c_x( u, v, w, a, b, c, j_cen) - F_d_x(u, v, w, a, b, c, j_cen)
    return gam_y

def function_alpha_z(j_cen : np.ndarray, a, b, c, i_cen : np.ndarray):
    u,v,w = i_cen[0], i_cen[1], i_cen[2]
    alph_z = F_e_y( u, v, w, a, b, c, j_cen) - F_f_y( u, v, w, a, b, c, j_cen)
    return alph_z

def function_beta_z(j_cen : np.ndarray, a, b, c, i_cen : np.ndarray):
    u,v,w = i_cen[0], i_cen[1], i_cen[2]
    beta_z = F_e_x(u, v, w, a, b, c, j_cen) - F_f_x(u, v, w, a, b, c, j_cen)
    return beta_z

def function_gamma_z(j_cen : np.ndarray, a, b, c, i_cen : np.ndarray, flag):
    u,v,w = i_cen[0], i_cen[1], i_cen[2]
    # print(G_a( u, v, w ,a, b, c, j_cen, flag))
    # print(G_b( u, v, w ,a, b, c, j_cen, flag))
    # print(G_c( u, v, w, a, b, c, j_cen, flag))
    # print(G_d( u, v, w, a, b, c, j_cen, flag))
    gam_z = G_a( u, v, w ,a, b, c, j_cen, flag) - G_b( u, v, w ,a, b, c, j_cen, flag) + G_c( u, v, w, a, b, c, j_cen, flag) - G_d( u, v, w, a, b, c, j_cen, flag)
    return gam_z

# 磁性体要素の中心の配列(n×3)を渡すと磁化ベクトルの係数行列(3n×3n)を返す
def function_M(cens : np.ndarray, a, b, c,flag):
    n = cens.shape[0]
    # print(n)
    cens_list = cens.tolist()
    m_list = []
    for i in range(n):
        i_cen = np.array(cens_list[i])
        for j in range(n):
            j_cen = np.array(cens_list[j])
            # print(j_cen)
            alph_x = function_alpha_x( j_cen, a, b, c, i_cen, flag)
            beta_x = function_beta_x(j_cen, a, b, c, i_cen)
            gamma_x = function_gamma_x(j_cen, a, b, c, i_cen)
            m_list.append(alph_x)
            m_list.append(beta_x)
            m_list.append(gamma_x)
        for k in range(n):
            k_cen = np.array(cens_list[k])
            alph_y = function_alpha_y( k_cen, a, b, c, i_cen)
            beta_y = function_beta_y(k_cen, a, b, c, i_cen, flag)
            gamma_y = function_gamma_y(k_cen, a, b, c, i_cen)
            m_list.append(alph_y)
            m_list.append(beta_y)
            m_list.append(gamma_y)
        for l in range(n):
            l_cen = np.array(cens_list[l])
            alph_z = function_alpha_z( l_cen, a, b, c, i_cen)
            beta_z = function_beta_z( l_cen, a, b, c, i_cen)
            gamma_z = function_gamma_z( l_cen, a, b, c, i_cen, flag)
            m_list.append(alph_z)
            m_list.append(beta_z)
            m_list.append(gamma_z)
    # print(len(m_list))
    m_np = np.array(m_list)
    m_np = m_np.reshape([3*n, 3*n])
    # print(m_np)
    # print(m_np.shape)
    return m_np




#################################################################################################################################
# 磁化ベクトルの値がおかしい
# z方向は全部正じゃないの??????????????????????????????????????????????????????????????????
#################################################################################################################################

# 磁化ベクトルの係数行列m(3n×3n)と磁性体要素中心の磁束密度の配列Bc(3n×1)と非透磁率の逆数を渡して磁性体の磁化ベクトルJ(3n×1)を出力
# ほんとにあってるのかい？？？？？？？？？？？？？？？？？？
def function_J( m : np.ndarray, Bc : np.ndarray, ur):
    j_eye = np.eye(m.shape[0])
    # print(((1-ur)/(4*np.pi))*m)
    # np.savetxt('M_v2.csv' ,((1-ur)/(4*np.pi))*m , delimiter=",")
    m_np_ = j_eye - ((1-ur)/(4*np.pi))*m
    # np.savetxt('M__.csv' , m_np_, delimiter=",")
    # print(m_np_)
    # print(np.linalg.inv(m_np_))
    # print(np.linalg.inv(m_np_).shape)
    # print(m_np_.shape)
    # print(np.linalg.inv(m_np_) @ m_np_)
    # np.savetxt('M_inv.csv' , np.linalg.inv(m_np_), delimiter=",")
    J = (1 - ur) * np.linalg.inv(m_np_) @ Bc
    return J

# 磁性体の情報(中心座標、サイズ),磁性体の分割数,コイルの情報(半径r[m],電流i[A])を与えると磁化ベクトル(n×3)を出力する関数
# u : 透磁率
def J_solver( cen : np.ndarray, w, d, h, n, m, l, r, i, u, flag):
    cens_list_np, a, b, c = make_elements(cen, w, d, h, n, m, l)
    # print(a)
    # print(b)
    # print(c)
    Bc = vector_Bc(cens_list_np, r, i, 1000)
    # print("Bc" ,Bc)
    M = function_M(cens_list_np, a, b, c, flag)
    # print("M " ,M)
    J = function_J(M, Bc, u)
    J = J.reshape(cens_list_np.shape)
    # print("J", J)
    return J


#################################################################################################################################
# 磁界導出のための係数行列生成

def function_Mm( cens : np.ndarray, a, b, c, p : np.ndarray, flag):
    n = cens.shape[0]
    cens_list = cens.tolist()
    m_list = []
    for j in range(n):
        j_cen = np.array(cens_list[j])
        alph_x = function_alpha_x( j_cen, a, b, c, p, flag)
        beta_x = function_beta_x(j_cen, a, b, c, p)
        gamma_x = function_gamma_x(j_cen, a, b, c, p)
        m_list.append(alph_x)
        m_list.append(beta_x)
        m_list.append(gamma_x)
    for k in range(n):
        k_cen = np.array(cens_list[k])
        alph_y = function_alpha_y( k_cen, a, b, c, p)
        beta_y = function_beta_y(k_cen, a, b, c, p, flag)
        gamma_y = function_gamma_y(k_cen, a, b, c, p)
        m_list.append(alph_y)
        m_list.append(beta_y)
        m_list.append(gamma_y)
    for l in range(n):
        l_cen = np.array(cens_list[l])
        alph_z = function_alpha_z( l_cen, a, b, c, p)
        beta_z = function_beta_z( l_cen, a, b, c, p)
        gamma_z = function_gamma_z( l_cen, a, b, c, p, flag)
        m_list.append(alph_z)
        m_list.append(beta_z)
        m_list.append(gamma_z)
        # print(gamma_z)
    m_np = np.array(m_list)
    m_np = m_np.reshape([3, 3*n])
    return m_np

def B_solver( cen : np.ndarray, w, d, h, n, m, l, r, i, u, p : np.array, flag1, flag_show):
    cens_list_np, a, b, c = make_elements(cen, w, d, h, n, m, l)
    Bc_vec = vector_Bc(cens_list_np, r, i, 1000)
    # print(Bc.shape)
    M = function_M(cens_list_np, a, b, c, False)
    if flag_show is True:
        print("M done!")
    # print(M.shape)
    J = function_J(M, Bc_vec, u)
    if flag_show is True:
        print("j is", J)
    Mm = function_Mm(cens_list_np, a, b, c, p, flag1)
    if flag_show is True:
        print("Mm done")
    Bc = Bc_solver(p, r, i, 1000)
    if flag_show is True:
        print("Bc is ",Bc)
    Bm = (Mm @ J) / (4*np.pi)
    Bm = Bm.T
    if Bm[0][0] is None:
        Bm[0][0] = 0
    if Bm[0][1] is None:
        Bm[0][1] = 0
    if Bm[0][2] is None:
        Bm[0][2] = 0
    if flag_show is True:
        print("Bm is ",Bm)
    B = Bm + Bc
    return B












