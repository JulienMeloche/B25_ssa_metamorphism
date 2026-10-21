import numpy as np

#script that calculate the optical diameter based on C13 in Crocus
#equation in Annexe B Vionnet et al 2012, from laws of Marbouty (1980)

phi = 1.0417e-9 # m s^-1
alpha = 1e-4

def h_rho(rho):

    if rho < 150:
        h = 1
    elif rho >= 150 and rho < 400:
        h = 1 - 0.004 * (rho - 150)
    elif rho >= 400:
        h = 0

    return h

def f_T(temperature):

    temp_C = temperature - 273.5
    if temp_C < -40:
        f = 0
    elif temp_C >= -40 and temp_C < -22:
        f =  0.011 * (temp_C + 40)
    elif temp_C >= -22 and temp_C < -6:
        f = 0.2 + 0.05 * (temp_C + 22)
        #f =  0.05 * (temp_C + 22)

    else:
        f = 0.7 - 0.05 * temp_C
    return f

def g_G_og(gradient):

    if gradient < 15:
        g = 0
    elif gradient >= 15 and gradient < 25:
        g = 0.01 * (gradient - 15)
    elif gradient >= 25 and gradient < 40:
        g = 0.1 + 0.037 * (gradient - 25)
    elif gradient >= 40 and gradient < 50:
        g = 0.65 + 0.02 * (gradient - 40)
    elif gradient >= 50 and gradient < 70:
        g = 0.85 + 0.0075 * (gradient - 50)
    else:
        g = 1
    return g

def g_G(gradient):

    if gradient < 15:
        g = 0
    elif gradient >= 15 and gradient < 25:
        g = 0.01 * (gradient - 15)
    elif gradient >= 25 and gradient < 40:
        g = (0.1 + 0.037 * (gradient - 25))
        if g > 0.3:
            g=0.3
    else:
        g = 0.3
    return g

def marbouty(temp, gradient, rho, og = False):
    if og == True:
        return f_T(temp) * g_G_og(gradient) * h_rho(rho) * phi 
    else:
        return f_T(temp) * g_G(gradient) * h_rho(rho) * phi 
    
def grad_s(temp, gradient, exponant = 0.4):
    if gradient <= 5:
        ds_dt = 11574.074 * np.exp(-6000/temp)
        #ds_dt = 2314.81 * np.exp(-6000/temp)
    if gradient >= 5:
        ds_dt = -2314.81 * np.exp(-6000/temp) * gradient**(exponant )
        
    return ds_dt

def grad_d(temp, gradient, rho, s, do, og = False, exponant = 0.4, sph_cond = False):
    #dendritic snow
    #XDENT1 = 2314.81
    #XVSPH2 = 11574.074
    if do < alpha * (4 - s):
        if gradient <= 5:
            if s == 1:
                dd_dt = alpha * (-2314.81 * np.exp(-6000/temp) * (s - 3))
            else:
                dd_dt = alpha * (-2314.81 * np.exp(-6000/temp) * (s - 3) + grad_s(temp, gradient) * (do/alpha -1)/(s - 3) )
        if gradient >= 5:
            if s == 0:
                dd_dt = alpha * (-2314.81 * np.exp(-6000/temp) * gradient**(exponant) * (s - 3))
            else:   
                dd_dt = alpha * (-2314.81 * np.exp(-6000/temp) * (s - 3) *  gradient**(exponant) + grad_s(temp, gradient) * (do/alpha -1)/(s - 3) )

    #non dendritic snow
    #eqn 4.12  these baron 2021
    else:
        if sph_cond:
            dd_dt = (1 + s)/2 * marbouty(temp, gradient, rho, og)

        elif (gradient <= 5):
            dd_dt = (do - 4 * alpha) / (1 + s) * grad_s(temp, gradient)
        elif (gradient > 5) and (gradient <= 15):
            dd_dt = (do - 4 * alpha) / (1 + s) * grad_s(temp, gradient)
        else:
            if s > 0:
                dd_dt = (1 + s)/2 * marbouty(temp, gradient, rho, og) + (do - 4 * alpha) / (1 + s) * grad_s(temp, gradient)
            else:
                dd_dt = (1 + s)/2 * marbouty(temp, gradient, rho, og)


    return dd_dt



def ssa_B25(ssa_i, T_mid, T_grad, density, dt, a_A = 1320, a_B = 1.52e-2, a_L = 1.09e-2, ssa_minimum = 0, a_tan = 0.3, alpha = 1e-1):
    """
    SSA evolution from Braun et al 2025
    doi:10.1017/jog.2024.109
    """

    #param fit 
    # a_A # sensitivity to temperature gradient
    # a_B # sensitivity to curvature
    # a_L # length of diffusion
    # ssa_minimum # ssa minimum value
    s_minimum = ssa_minimum * density
    s_i = ssa_i * density

    #constant
    D_v = 2e-5  #vapor diffusion coefficient
    #v_kin = 140  #kinetic velocity
    c_l =  1e-9  #capillary length
    rho_ice = 916.7 # density of ice
    kb = 1.380649e-23
    m_water = 2.99e-26 #kg , mass of water molecules

    v_kin = np.sqrt(kb * T_mid / (2 * np.pi * m_water))
    
    #convert dt in seconds
    dt_s = dt * 3600

    #temperature gradient term
    a_A = a_A * np.maximum(0, np.tanh(a_tan * (ssa_i - ssa_minimum)))
    TG_contribution = a_A * beta(T_mid) * T_grad

    sigma_H = (ssa_i * rho_ice)**2

    #isothermal term
    isothermal_contribution = a_B * sigma_H**1.5 * compute_rho_vs(T_mid) * c_l

    #transition coefficient between kinetic limited to diffusion limited
    KL_DL_transition = 1 + D_v / (alpha * v_kin * a_L/s_i)

    # variation of ssa over time
    ds_dt =  - 2 * D_v * s_i / rho_ice * (TG_contribution + isothermal_contribution )/ KL_DL_transition 
    s_f = s_i + ds_dt * dt_s

    return s_f / density

def compute_rho_vs(T_mid, derivative = False):
    #constant
    m_water = 2.99e-26 #kg , mass of water molecules
    kb = 1.380649e-23 #boltzman constant
    P_0 = 611 # Pa
    T_0 = 273.15 #K
    L_fus = 8.37e-20 # J
    if derivative:
        return m_water/(kb * T_mid**3) * P_0 * (L_fus/kb - T_mid) * np.exp(L_fus/kb * (1/T_0 - 1/T_mid))
    else:
        return m_water/(kb * T_mid) * P_0 * np.exp(L_fus/kb * (1/T_0 - 1/T_mid))

def beta(T_mid):
    return compute_rho_vs(T_mid, derivative = True)
