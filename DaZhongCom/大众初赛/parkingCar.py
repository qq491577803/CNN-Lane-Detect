import numpy as np

def parking(Wcar,Rmin,Lfb,Lf,Lb,):
    '''
    计算泊车时开始位置，以及泊车两个阶段车轮转过的距离
    :param Wcar:车体的宽度尺寸
    :param Rmin:方向盘旋转至最大角度时，内侧车轮转弯半径
    :param Lfb:前后轮轴之间的尺寸
    :param Lf:前车轴距离车前身的尺寸
    :param Lb:车后轴距离车后身的尺寸
    :return:dict
            ：N1N4:开始泊车时车体和目标纵距离
            ：Wd:开始泊车时车体和目标横向距离
            :NoN3:初始轨迹车轮转过长度
            :N3N4:最后反向圆弧车轮转过的长度
    '''
    '''第一阶段：车体和目标车位横向距离和纵向距离计算'''
    Wd = (2 * Rmin + Wcar) * (1 - np.cos(np.arcsin((np.sqrt((np.square(Rmin + Wcar)) + (np.square(Lfb + Lf)) - np.square(Rmin)) + Lb) / (2 * Rmin + Wcar)))) - Wcar
    delta = np.arccos(1 - (Wcar+Wd)/(2*Rmin+Wcar))
    N1N4 = (2 * Rmin+Wcar)* np.sin(delta)

    """第二阶段：初始轨迹圆弧长度"""
    NoN3 = Rmin * delta
    '''第三阶段：最后圆弧长度'''
    N3N4 =(Rmin + Wcar) * delta
    pars = {"N1N4":N1N4 ,"Wd":Wd,"NoN3":NoN3,"N3N4":N3N4}
    return pars