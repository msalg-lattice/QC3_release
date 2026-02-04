import numpy as np, sys, math
pi=np.pi; LA=np.linalg; exp=np.exp
from scipy.linalg import block_diag
from scipy.special import erfi
import numba
from numba import njit

import defns
from constants import *
sqrt = defns.sqrt
check_real = defns.check_real

#This is an asymptotic expansion of erfc function. Numba doesn't accept scipy.special.erfc
@njit(fastmath=True,nogil=True,cache=True)
def myerfc2(x):
    return exp(-x**2)/np.sqrt(pi)/x*(1.- 1./2/x**2 + 3./(2.*x**2)**2)

@njit(fastmath=True,cache=True)
def myerfi(z):
    """
    Fast nopython path for nearly-pure-imag inputs using erfi(i*y)=i*erf(y).
    For general complex z, fall back to SciPy.erfi inside an objmode block.
    (Objmode calls are slower — use only when necessary.)
    """
    tol = 1e-15
    if abs(z.real) <= tol * max(1.0, abs(z.imag)):
        # exact identity for pure imaginary
        return 1j * math.erf(z.imag)

    # otherwise call SciPy.erfi in object mode
    res = 0+0j
    # Note: numba.objmode only allowed inside an njit function
    # and will execute the block in object mode (Python/SciPy)
    with numba.objmode(res='complex128'):
        res = erfi(z)   # SciPy does the robust Faddeeva-based computation
    return res

################################################################################
# Find maximum n needed in UV regime for sum_smooth_nnk & sum_full_nnk
################################################################################
@njit(fastmath=True,cache=True)
def getnmaxreal(cutoff,hhk,gam,x2):
  alpha = get_alpha()
  n0=5
  res = hhk*gam * 2*pi*np.sqrt(pi/alpha) * exp(alpha*x2)*myerfc2(np.sqrt(alpha)*n0)
  while(res>cutoff):
    n0+=1
    res = hhk*gam * 2*pi*np.sqrt(pi/alpha) * exp(alpha*x2)*myerfc2(np.sqrt(alpha)*n0)

  nmax_real = n0*gam
  # print('hhk*gam, nmax_real:',hhk*gam,nmax_real)
  return nmax_real
################################################################################
# Compute summand, modulo a common overall constant factor
################################################################################
@njit(fastmath=True,cache=True)
def summand(x2, rvec, l1,m1,l2,m2):
  r2 = sum(rvec**2)
  prop_den = x2-r2
  # const = hhk/(16*pi**2*L**4*omk*E2k)   # put in at end
  UV_term = exp(get_alpha()*prop_den)
  Ylm_term = defns.ylm(rvec,l1,m1) * defns.ylm(rvec,l2,m2) # no q (put 2pi/L in separately)
  return UV_term * Ylm_term * 1/prop_den  #*const
################################################################################
# Compute FULL sum w/o splitting pole/smooth parts
################################################################################
@njit(fastmath=True,cache=True)
def sum_full_nnk(E,nnP,L,nnk, Mijk=np.array([1,1,1]), waves='sp'):
  [Mi, Mj, Mk] = Mijk
  W = defns.get_lm_size(waves)
  twopibyL = 2*pi/L

  kvec = nnk*twopibyL
  omk = check_real(sqrt(np.sum(kvec**2)+Mi**2))
  E2k = E-omk
  nnP2k = nnP-nnk
  sig_i = E2k**2 - np.sum(nnP2k**2)*twopibyL**2
  q2_i = defns.lambda_tri(sig_i,Mj**2,Mk**2)/(4*sig_i)
  alpha_ij = check_real(sqrt(q2_i + Mj**2) / sqrt(sig_i))

  hhk = defns.hh(sig_i, Mjk=[Mj,Mk])
  if hhk==0:
    return np.zeros((W,W))

  gam = E2k/check_real(sqrt(sig_i))
  x2 = q2_i/twopibyL**2

  cutoff = get_cutoff()
  nmax_real = getnmaxreal(cutoff,hhk,gam,x2)
  nmax = int(np.ceil(nmax_real))

  out = np.zeros((W,W))
  for n1 in range(-nmax,nmax+1):
    for n2 in range(-nmax,nmax+1):
      for n3 in range(-nmax,nmax+1):
        nna = np.array([n1,n2,n3], dtype=np.float64)
        if LA.norm(nna)<nmax_real: # and list(nna) not in nna_on_list:
          if np.all(nnP2k == 0):
            rvec = nna
          else:
            nnP2k_ = np.asarray(nnP2k, dtype=np.float64)
            rvec = nna + nnP2k_ * (np.dot(nna,nnP2k_)/np.sum(nnP2k_**2) * (1/gam-1) - alpha_ij/gam)
          for i1 in range(W):
            [l1,m1] = defns.lm_idx(i1, waves=waves)
            for i2 in range(W):
              [l2,m2] = defns.lm_idx(i2, waves=waves)
              out[i1,i2] += (2*pi/L)**(l1+l2) * summand(x2,rvec, l1,m1,l2,m2)
  return out # *const


################################################################################
# Compute PV integral
################################################################################
@njit(fastmath=True,cache=True)
def int_nnk(L,gam,x2,waves='sp'):
  alpha = get_alpha()
  W = defns.get_lm_size(waves)

  twopibyL = 2*pi/L
  x = sqrt(x2)
  ax2 = alpha*x2
  e_ax2 = exp(ax2)
  erfi_sqrt_ax2 = myerfi(sqrt(ax2))
  c = 4*pi*gam

  out_dict = {}
  if 's' in waves:
    factor1 = -sqrt(pi/alpha)*0.5*e_ax2
    factor2 = 0.5*pi*x*erfi_sqrt_ax2
    out_dict[0] = c*(factor1 + factor2)
  if 'p' in waves:
    factor1 = -sqrt(pi/alpha**3)*(1+2*ax2)*e_ax2/4
    factor2 = 0.5*pi*x**3*erfi_sqrt_ax2
    out_dict[1] = twopibyL**2 * c * (factor1 + factor2) # no q factors
  if 'd' in waves:
    factor1 = -sqrt(pi/alpha**5)*(3+2*ax2+4*ax2**2)*e_ax2/8
    factor2 = 0.5*pi*x**5*erfi_sqrt_ax2
    out_dict[2] = twopibyL**4 * c * (factor1 + factor2) # no q factors

  out = np.zeros((W,W))
  for i in range(W):
    l, _ = defns.lm_idx(i,waves=waves)
    if abs(out_dict[l].imag)>1e-13:
      print(out_dict[l])
      raise ValueError('Error in F_fast: imaginary part encountered')
    out[i,i] = out_dict[l].real
  return out

################################################################################
# Full \wt{F}^{(i)}(k) matrix w/o splitting pole/smooth terms
################################################################################
@njit(fastmath=True,cache=True)
def F_i_nnk(E,nnP,L,nnk, Mijk=np.array([1,1,1]), waves='sp'):
  [Mi,Mj,Mk] = Mijk

  twopibyL = 2*pi/L
  Pvec = nnP*twopibyL
  kvec = nnk*twopibyL
  omk = check_real(sqrt(np.sum(kvec**2)+Mi**2))
  E2k = E - omk

  sig_i = check_real(defns.sigma_i(E,Pvec,kvec,Mi=Mi))
  q2_i = defns.lambda_tri(sig_i,Mj**2,Mk**2)/(4*sig_i)
  hhk = defns.hh(sig_i, Mjk=[Mj,Mk])

  gam = E2k/check_real(sqrt(sig_i))
  x2 = q2_i/twopibyL**2


  # const = 1/(32*pi**2*L*omk*E2k)
  const = hhk/(16*pi**2*L**4*omk*E2k) # Assumes no flavor symmetry; divide by 2 for symmetric case
                                      # Convention is L for beyond_iso, L**4 for TOPT papers
  sum_bare = sum_full_nnk(E,nnP,L,nnk, Mijk=Mijk, waves=waves)
  int_bare = int_nnk(L,gam,x2,waves=waves)
  # print(sum_bare)
  # print(int_bare)
  return const*(sum_bare-int_bare)

################################################################################
# Full \wt{F}^{(i)} matrix computed from scratch
################################################################################
def F_i_full_scratch(E,nnP,L, Mijk=np.array([1,1,1]), waves='sp', nnk_list=None, diag_only=True):
  if nnk_list==None:
    nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=Mijk)
  Fi_diag = []
  for nnk in nnk_list:
    nnP_array = np.asarray(nnP)
    nnk_array = np.asarray(nnk)
    Fi_diag.append(F_i_nnk(E,nnP_array,L,nnk_array, Mijk=Mijk, waves=waves))
  if diag_only==True:
    return Fi_diag
  else:
    return block_diag(*Fi_diag)

################################################################################
# Full 2+1 matrix \wt{F} computed from scratch
################################################################################
def F_full_2plus1_scratch(E,nnP,L, M12=np.array([1,1]), waves='sp', nnk_lists_12=None, diag_only=True):
  M1, M2 = M12
  #M123 = [M1, M1, M2]
  if nnk_lists_12 == None:
    nnk_lists_12 = [defns.list_nnk_nnP(E,L,nnP, Mijk=[M1,M1,M2])]
    nnk_lists_12 += [defns.list_nnk_nnP(E,L,nnP, Mijk=[M2,M1,M1])]
  F1_diag = F_i_full_scratch(E,np.array(nnP),L, Mijk=[M1,M1,M2], waves=waves, nnk_list=nnk_lists_12[0], diag_only=True)
  F2_diag = F_i_full_scratch(E,np.array(nnP),L, Mijk=[M2,M1,M1], waves='s', nnk_list=nnk_lists_12[1], diag_only=True)
  F_diag = F1_diag + F2_diag
  if diag_only==True:
    return F_diag
  else:
    return block_diag(*F_diag)

################################################################################
# Full ND matrix \wt{F} computed from scratch
################################################################################
def F_full_ND_scratch(E,nnP,L, M123=np.array([1,1,1]), waves='s', nnk_lists_123=None, diag_only=True):
  F_diag = []
  for i in range(3):
    Mijk = defns.get_Mijk(M123,i)
    if nnk_lists_123 == None:
      nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=Mijk)
    else:
      nnk_list = nnk_lists_123[i]
    Fi_diag = F_i_full_scratch(E,np.array(nnP),L, Mijk=Mijk, waves=waves, nnk_list=nnk_list, diag_only=True)
    F_diag += Fi_diag
  if diag_only==True:
    return F_diag
  else:
    return block_diag(*F_diag)


################################################################################
# Full ID (identical particles) matrix \wt{F} computed from scratch
################################################################################
def F_full_ID_scratch(E,nnP,L, nnk_list=None, diag_only=True):
  waves = 's'   # only use s-wave for ID case (Kdf3 isn't ready for d-wave)
  if nnk_list == None:
    nnk_list = defns.list_nnk_nnP(E,L,nnP)
  # Divide by 2 for Bose symmetry
  F_diag = F_i_full_scratch(E,np.array(nnP),L, Mijk=np.array([1,1,1]), waves=waves, nnk_list=nnk_list, diag_only=True)
  if diag_only==True:
    return 0.5*np.array(F_diag)
  else:
    return 0.5*block_diag(*F_diag)


################################################################################
# Full 2-pt. F matrices for ND & ID cases
################################################################################
# ND case
@njit(fastmath=True,cache=True)
def F_2pt_ND(E2,nnP2,L, M12=np.array([1,1]), waves='sp'):
  M1,M2 = M12
  F = 2*L**3 * F_i_nnk(E2+1,nnP2,L,np.array([0,0,0]), Mijk=np.array([1,M1,M2]), waves=waves)
  return F

# ID case (s-wave only)
@njit(fastmath=True,cache=True)
def F_2pt_ID(E2,nnP2,L, M12=np.array([1,1])):
  M1,M2 = M12
  F = L**3 * F_i_nnk(E2+1,nnP2,L,np.array([0,0,0]), Mijk=np.array([1,1,1]), waves='s')
  return F
