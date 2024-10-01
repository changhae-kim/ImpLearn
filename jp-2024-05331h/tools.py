import numpy

from ase import Atoms
from scipy.special import logsumexp


def calc_Kg(T, G_c, G_d, g_I, G_I, g_II, G_II, g_III, G_III, g_IV, G_IV):

    kB = 1.380649e-23 / 4.3597447222071e-18

    g_I = numpy.array(g_I)
    G_I = numpy.array(G_I)
    g_II = numpy.array(g_II)
    G_II = numpy.array(G_II)
    g_III = numpy.array(g_III)
    G_III = numpy.array(G_III)
    g_IV = numpy.array(g_IV)
    G_IV = numpy.array(G_IV)

    log_K1 = numpy.log(g_I) - (G_I+2*G_c)/(kB*T)
    log_K2 = numpy.log(g_II) - (G_II+G_c+G_d)/(kB*T)
    log_K3 = numpy.log(g_III) - (G_III+2*G_d)/(kB*T)
    log_K4 = numpy.log(g_IV) - (G_IV+G_c+2*G_d)/(kB*T)

    log_K12 = logsumexp(log_K2) - logsumexp(log_K1)
    log_K23 = logsumexp(log_K3) - logsumexp(log_K2)
    log_K34 = logsumexp(log_K4) - logsumexp(log_K3)

    return numpy.exp(log_K12), numpy.exp(log_K23), numpy.exp(log_K34)

def calc_k2_K1(T, G_a, g_s, G_s, g_r, G_r, g_ts, G_ts):

    kB = 1.380649e-23 / 4.3597447222071e-18
    h = 6.62607015e-34 / 4.3597447222071e-18

    g_s = numpy.array(g_s)
    G_s = numpy.array(G_s)
    g_r = numpy.array(g_r)
    G_r = numpy.array(G_r)
    g_ts = numpy.array(g_ts)
    G_ts = numpy.array(G_ts)

    log_k2_K1_K0 = numpy.log(kB*T/h) + numpy.log(g_ts) - (G_ts)/(kB*T)
    log_K1_K0 = numpy.log(g_r) - (G_r)/(kB*T)
    log_K0 = numpy.log(g_s) - (G_s+G_a)/(kB*T)

    log_k2 = logsumexp(log_k2_K1_K0) - logsumexp(log_K1_K0)
    log_K1 = logsumexp(log_K1_K0) - logsumexp(log_K0)

    return numpy.exp(log_k2), numpy.exp(log_K1)

def calc_kg(T, g_r, G_r, g_ts, G_ts):

    kB = 1.380649e-23 / 4.3597447222071e-18
    h = 6.62607015e-34 / 4.3597447222071e-18

    g_r = numpy.array(g_r)
    G_r = numpy.array(G_r)
    g_ts = numpy.array(g_ts)
    G_ts = numpy.array(G_ts)

    log_kg_K0 = numpy.log(kB*T/h) + numpy.log(g_ts) - (G_ts)/(kB*T)
    log_K0 = numpy.log(g_r) - (G_r)/(kB*T)

    log_kg = logsumexp(log_kg_K0) - logsumexp(log_K0)

    return numpy.exp(log_kg)

def calc_ac_rate(T, P, k2, K1):

    R = 8.31446261815324 # J/K*mol
    atm = 1.01325e+5 # Pa
    rho = 0.921 # g/cm3
    m_a = 28.054 # g/mol

    S_a = 1.55e-4 * numpy.exp(+(5.89e+3)/(R*T)) # g/g*atm
    S_a = (rho*1e+3)/(m_a) * S_a # mol/L*atm

    c_inf = S_a*P # mol/L
    c_0 = (atm)/(R*T)*1e-3 # mol/dm^3

    rate = (k2*K1*c_inf/c_0) / (1.0 + K1*c_inf/c_0) # s^-1

    return rate

def calc_dc_rate(T, P):

    R = 8.31446261815324 # J/K*mol
    N = 6.02214076e+23 # mol^-1
    atm = 1.01325e+5 # Pa
    rho = 0.921 # g/cm3
    m_a = 28.054 # g/mol
    R_a = 2.444 # Ang

    D_a = 5.08e-4 * numpy.exp(-(3.78e+4)/(R*T)) # m^2/s
    S_a = 1.55e-4 * numpy.exp(+(5.89e+3)/(R*T)) # g/g*atm
    S_a = (rho*1e+3)/(m_a) * S_a # mol/L*atm

    k_D = 2.0*numpy.pi * (D_a*1e+2) * (R_a*1e-9) * N # dm^3/mol*s

    c_inf = S_a*P # mol/L

    rate = k_D*c_inf # s^-1

    return rate

def calc_pdc_rate(T, P, k2, K1):

    R = 8.31446261815324 # J/K*mol
    N = 6.02214076e+23 # mol^-1
    atm = 1.01325e+5 # Pa
    rho = 0.921 # g/cm3
    m_a = 28.054 # g/mol
    R_a = 2.444 # Ang

    D_a = 5.08e-4 * numpy.exp(-(3.78e+4)/(R*T)) # m^2/s
    S_a = 1.55e-4 * numpy.exp(+(5.89e+3)/(R*T)) # g/g*atm
    S_a = (rho*1e+3)/(m_a) * S_a # mol/L*atm

    k_D = 2.0*numpy.pi * (D_a*1e+2) * (R_a*1e-9) * N # dm^3/mol*s

    c_inf = S_a*P # mol/L
    c_0 = (atm)/(R*T)*1e-3 # mol/dm^3
    c_a = 0.5 * (c_inf - c_0/K1 - k2/k_D) + (0.25 * (c_inf - c_0/K1 - k2/k_D)**2.0 + c_inf*c_0/K1)**0.5 # mol/L

    rate = (k2*K1*c_a/c_0) / (1.0 + K1*c_a/c_0) # s^-1

    return rate

def get_sxc(coords):

    atm = 'H' * 6
    xyz = numpy.empty((6, 3))
    xyz[:4] = coords[:]
    xyz[4] = 0.5 * (coords[0] + coords[1])
    xyz[5] = 0.5 * (coords[2] + coords[3])
    mol = Atoms(atm, xyz)

    sxc = [
            mol.get_distance(4, 5),
            mol.get_distance(0, 1) + mol.get_distance(2, 3),
            mol.get_distance(0, 1) - mol.get_distance(2, 3),
            mol.get_angle(0, 4, 5) + mol.get_angle(4, 5, 3),
            mol.get_angle(0, 4, 5) - mol.get_angle(4, 5, 3),
            mol.get_dihedral(0, 4, 5, 3),
            ]

    if sxc[-1] > 180.0:
        sxc[-1] = sxc[-1] - 360.0

    return sxc

