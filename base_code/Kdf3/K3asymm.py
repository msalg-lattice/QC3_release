import pickle
from pathlib import Path
import numpy as np
from numba import njit
pi=np.pi; LA=np.linalg;
import defns
sqrt = defns.sqrt

_CG_TABLE_PATH = Path(__file__).resolve().parent / 'cg_table.pkl'
with _CG_TABLE_PATH.open('rb') as f:
    CG_TABLE = pickle.load(f)


def _build_dense_cg_table(cg_table):
    max_ell = 0
    max_S = 0
    max_J = 0
    max_abs_m = 0
    for (ell, m_ell, S, m, J, mJ), coeff in cg_table.items():
        _ = coeff
        max_ell = max(max_ell, ell)
        max_S = max(max_S, S)
        max_J = max(max_J, J)
        max_abs_m = max(max_abs_m, abs(m_ell), abs(m), abs(mJ))

    mshift = max_abs_m
    dense = np.zeros(
        (max_ell + 1, 2 * mshift + 1, max_S + 1, 2 * mshift + 1, max_J + 1, 2 * mshift + 1),
        dtype=np.float64,
    )
    for (ell, m_ell, S, m, J, mJ), coeff in cg_table.items():
        dense[ell, m_ell + mshift, S, m + mshift, J, mJ + mshift] = float(np.real(coeff))
    return dense, mshift


CG_TABLE_DENSE, CG_M_SHIFT = _build_dense_cg_table(CG_TABLE)

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
    return np.sum([CG_TABLE[ell, m_ell, S, m, J, mJ] * defns.ylm(-kvec_hat, ell, m_ell) for m_ell in range(-ell, ell+1)])

def ZZ_angular(J, ell_prime, S_prime, m_prime, ell, S, m, kvec_prime_hat, kvec_hat):
    return np.sum([Z_angular(J, mJ, ell_prime, S_prime, m_prime, kvec_prime_hat) * np.conj(Z_angular(J, mJ, ell, S, m, kvec_hat)) for mJ in range(-J, J+1)])


@njit(fastmath=True, cache=True)
def Z_angular_fast(J, mJ, ell, S, m, kvec_hat, cg_table_dense, cg_m_shift):
    out = 0.0 + 0.0j
    for m_ell in range(-ell, ell + 1):
        cg = cg_table_dense[ell, m_ell + cg_m_shift, S, m + cg_m_shift, J, mJ + cg_m_shift]
        out += cg * defns.ylm(-kvec_hat, ell, m_ell)
    return out


@njit(fastmath=True, cache=True)
def ZZ_angular_fast(J, ell_prime, S_prime, m_prime, ell, S, m, kvec_prime_hat, kvec_hat, cg_table_dense, cg_m_shift):
    out = 0.0 + 0.0j
    for mJ in range(-J, J + 1):
        zprime = Z_angular_fast(J, mJ, ell_prime, S_prime, m_prime, kvec_prime_hat, cg_table_dense, cg_m_shift)
        z = Z_angular_fast(J, mJ, ell, S, m, kvec_hat, cg_table_dense, cg_m_shift)
        out += zprime * np.conj(z)
    return out

def K3mat_ID_asymm(E, L, nnP, Jmax, waves, hfunc, Ktildefunc, nnk_list=None):
    twopibyL = 2*pi/L
    Pvec = twopibyL * np.array(nnP)
    if nnk_list==None:
        nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=[1,1,1])

    Ecm = sqrt(E**2 - np.sum(Pvec**2))

    wave_l = [get_l_from_wave(wave) for wave in waves]
    wave_offsets = []
    offset = 0
    for l in wave_l:
        wave_offsets.append(offset)
        offset += 2*l + 1

    N = len(nnk_list)
    M = offset

    kvec_hat_list = []
    knorm_list = []
    qstar_list = []
    hfunc_q2_list = []
    for nnk in nnk_list:
        kvec = twopibyL * np.array(nnk)
        qstar = sqrt(defns.qst2_i(E, Pvec, kvec))
        kvec_hat_list.append(defns.vec_hat(kvec))
        knorm_list.append(LA.norm(kvec))
        qstar_list.append(qstar)
        hfunc_q2_list.append(hfunc(qstar**2))

    K3mat = np.zeros((N, M, N, M), dtype=np.complex128)
    for i_nnk in range(N):
        kvec_prime_hat = kvec_hat_list[i_nnk]
        knorm_prime = knorm_list[i_nnk]
        qstar_prime = qstar_list[i_nnk]
        hfunc_q2_prime = hfunc_q2_list[i_nnk]

        for i_wave, S_prime in enumerate(wave_l):
            for m_prime in range(-S_prime, S_prime+1):
                iK3mat_col = wave_offsets[i_wave] + (m_prime + S_prime)
                qpow_prime = qstar_prime**S_prime

                for j_nnk in range(N):
                    kvec_hat = kvec_hat_list[j_nnk]
                    knorm = knorm_list[j_nnk]
                    qstar = qstar_list[j_nnk]
                    hfunc_q2 = hfunc_q2_list[j_nnk]

                    for j_wave, S in enumerate(wave_l):
                        for m in range(-S, S+1):
                            iK3mat_row = wave_offsets[j_wave] + (m + S)
                            qpow = qstar**S

                            elem = 0.0 + 0.0j
                            for J in range(0, Jmax + 1):
                                for ell_prime in range(abs(J - S_prime), J + S_prime + 1):
                                    h_prime = knorm_prime**ell_prime * qpow_prime * hfunc_q2_prime
                                    for ell in range(abs(J - S), J + S + 1):
                                        h = knorm**ell * qpow * hfunc_q2
                                        Ktilde = Ktildefunc(Ecm, J, ell_prime, S_prime, ell, S)
                                        radial = h_prime * Ktilde * h
                                        angular = ZZ_angular_fast(
                                            J, ell_prime, S_prime, m_prime, ell, S, m,
                                            kvec_prime_hat, kvec_hat, CG_TABLE_DENSE, CG_M_SHIFT
                                        )
                                        elem += radial * angular
                            K3mat[i_nnk, iK3mat_col, j_nnk, iK3mat_row] = elem

    K3mat = K3mat.reshape((N*M, N*M))
    return K3mat
