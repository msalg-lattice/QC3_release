import numpy as np
pi=np.pi; LA=np.linalg;
from sympy.physics.quantum.cg import CG
import defns
sqrt = defns.sqrt

def get_l_from_wave(wave):
    wave_ = wave[0].lower()
    if wave_=='s':
        return 0
    elif wave_=='p':
        return 1
    elif wave_=='d':
        return 2
    elif wave_=='f':
        return 3
    else:
        raise ValueError("Unknown wave type: {}".format(wave))

def Z_angular(J, mJ, ell, S, m, kvec_hat):
    return np.sum([CG(ell, m_ell, S, m, J, mJ).doit().n() * defns.ylm(-kvec_hat, ell, m_ell) for m_ell in range(-ell, ell+1)])

def ZZ_angular(J, ell_prime, S_prime, m_prime, ell, S, m, kvec_prime_hat, kvec_hat):
    return np.sum([Z_angular(J, mJ, ell_prime, S_prime, m_prime, kvec_prime_hat) * np.conj(Z_angular(J, mJ, ell, S, m, kvec_hat)) for mJ in range(-J, J+1)])

def K3_ID_asymm_angular_basis(E, Pvec, J, ell_prime, S_prime, ell, S, k_prime, k, hfunc, Ktildefunc):
    Ecm = sqrt(E**2-sum([x**2 for x in Pvec]))
    qstar_k_prime = sqrt(defns.qst2_i(E, Pvec, k_prime))
    qstar_k = sqrt(defns.qst2_i(E, Pvec, k))
    h_prime = LA.norm(k_prime)**ell * qstar_k_prime**S_prime * hfunc(qstar_k_prime**2)
    h = LA.norm(k)**ell * qstar_k**S * hfunc(qstar_k**2)
    Ktilde = Ktildefunc(Ecm, J, ell_prime, S_prime, ell, S)
    return h_prime * Ktilde * h

def K3mat_ID_asymm(E, L, nnP, Jmax, waves, hfunc, Ktildefunc, nnk_list=None):
    Pvec = 2*pi/L * np.array(nnP)
    if nnk_list==None:
        nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=[1,1,1])

    N = len(nnk_list)
    M = sum(2*get_l_from_wave(wave) + 1 for wave in waves)
    K3mat = np.zeros((N, M, N, M))
    for i_nnk, nnk_prime in enumerate(nnk_list):
        kvec_prime = 2*pi/L * np.array(nnk_prime)
        for i_wave, wave_prime in enumerate(waves):
            S_prime = get_l_from_wave(wave_prime)
            for m_prime in range(-S_prime, S_prime+1):
                iK3mat_col = sum(2*get_l_from_wave(waves[j]) + 1 for j in range(i_wave)) + (m_prime + S_prime)
                for j_nnk, nnk in enumerate(nnk_list):
                    kvec = 2*pi/L * np.array(nnk)
                    for j_wave, wave in enumerate(waves):
                        S = get_l_from_wave(wave)
                        for m in range(-S, S+1):
                            iK3mat_row = sum(2*get_l_from_wave(waves[j]) + 1 for j in range(j_wave)) + (m + S)
                            K3mat[i_nnk, iK3mat_col, j_nnk, iK3mat_row] = np.sum([[[K3_ID_asymm_angular_basis(E, Pvec, J, ell_prime, S_prime, ell, S, kvec_prime, kvec, hfunc, Ktildefunc) * ZZ_angular(J, ell_prime, S_prime, m_prime, ell, S, m, defns.vec_hat(kvec_prime), defns.vec_hat(kvec)) for ell in range(abs(J - S), J + S + 1)] for ell_prime in range(abs(J - S_prime), J + S_prime + 1)] for J in range(0, Jmax + 1)])

    K3mat = K3mat.reshape((N*M, N*M))
    return K3mat
