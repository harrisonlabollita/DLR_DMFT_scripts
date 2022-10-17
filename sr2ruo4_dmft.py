# DMFT calculation for Sr2RuO4 using parameter free tail-fitting

import numpy as np
from triqs.sumk import *
from triqs.gf import *
import triqs.utility.mpi as mpi
from triqs.utility.dichotomy import dichotomy
from triqs_cthyb import *
from h5 import HDFArchive
from triqs.operators import util
from triqs.lattice.utils import k_space_path, TB_from_wannier90
from triqs.operators import c_dag, c, Operator

from copy import deepcopy
from timeit import default_timer as timer

np.set_printoptions(precision=6,suppress=True)

# wannier90 helper functions 
def load_data_generic(path, name='w2w'):
    hopping, num_wann = parse_hopping_from_wannier90_hr_dat(path + name +'_hr.dat')
    units = parse_lattice_vectors_from_wannier90_wout(path + name +'.wout')
    return hopping, units, num_wann

def get_TBL(path, name='w2w', extend_to_spin=False, add_local=None, add_field=None, renormalize=None):

    hopping, units, num_wann = load_data_generic(path, name=name)
    if extend_to_spin:
    	hopping, num_wann = extend_wannier90_to_spin(hopping, num_wann)
    if add_local is not None:
        hopping[(0,0,0)] += add_local
    if renormalize is not None:
        assert len(np.shape(renormalize)) == 1, 'Give Z as a vector'
        assert len(renormalize) == num_wann, 'Give Z as a vector of size n_orb (times two if SOC)'
        
        Z_mat = np.diag(np.sqrt(renormalize))
        for R in hopping:
            hopping[R] = np.dot(np.dot(Z_mat, hopping[R]), Z_mat)

    if add_field is not None:
        hopping[(0,0,0)] += add_field

    TBL = TBLattice(units = units, hopping = hopping, orbital_positions = [(0,0,0)]*num_wann,
                    orbital_names = [str(i) for i in range(num_wann)])
    return TBL

# old function to extract density for a given mu, to be used by dichotomy function to determine mu
def Density(mu):
    # calling the k sum here
    # github.com/TRIQS/triqs/blob/3.0.x/python/triqs/sumk/sumk_discrete.py#L73
    dens =  sumk(mu = mu, Sigma = S.Sigma_iw).total_density()
    if abs(dens.imag) > 1e-20:
            mpi.report("Warning: Imaginary part of density will be ignored ({})".format(str(abs(dens.imag))))
    return dens.real

def fit_dlr(gf, **kwargs):
    """
    Obtain Discrete Lehemann Representation (DLR) of imaginary time Green's function.
    
    This is a wrapper to the libdlr library: https://github.com/jasonkaye/libdlr
    """
    from pydlr import dlr, kernel

    if isinstance(gf, BlockGf):
        gf_ = gf.copy()
        for block, g in gf_:
            g.data[:,:,:] = fit_dlr(g, **kwargs)
        return gf_

    is_gf = isinstance(gf, Gf)
    is_mesh = (isinstance(gf.mesh, MeshImTime) or isinstance(gf.mesh, MeshImFreq))

    assert is_gf and is_mesh, "fit_dlr expects imaginary time or Matsubara Green's function objects."

    # construct DLR basis
    dlr_basis = dlr(**kwargs)

    if isinstance(gf.mesh, MeshImTime):
        tau = np.array([x.real for x in gf.mesh], dtype=float)
        beta = gf.mesh.beta
        Gdlr = dlr_basis.lstsq_dlr_from_tau(tau, gf.data, beta)
        return dlr_basis.eval_dlr_tau(Gdlr, tau, beta)

    if isinstance(gf.mesh, MeshImFreq):
        iwn= np.array([x.real + 1j*x.imag for x in gf.mesh], dtype=complex)
        beta = gf.mesh.beta
        Gdlr = dlr_basis.lstsq_dlr_from_matsubara(iwn, gf.data, beta)
        return dlr_basis.eval_dlr_freq(Gdlr, iwn, beta)

def calculate_analytic_moments(dm, hdiag, gf_struct, h_int):
    
    def comm(A,B): return A*B - B*A
    def anticomm(A,B): return A*B + B*A
   
    # the hartree term
    sigma_hf = {bl : np.zeros((bl_size,bl_size)) for bl, bl_size in gf_struct}
    for bl, bl_size in gf_struct:
        for orb1 in range(bl_size):
            for orb2 in range(bl_size):
                op = -anticomm(comm(h_int,c(bl,orb1)),c_dag(bl,orb2)) # ΣHF = -{[H,c],c+}
                
                for term, coef in op:
                    
                    bl1,u1 = term[0][1]
                    bl2,u2 = term[1][1]
                    
                    sigma_hf[bl][orb1,orb2] += coef*trace_rho_op(dm, c_dag(bl1,u1)*c(bl2,u2),hdiag)
    
    
    # the iωn term
    sigma_1 = {bl : np.zeros((bl_size,bl_size)) for bl, bl_size in gf_struct}
    for bl,bl_size in gf_struct:
        for orb1 in range(bl_size):
            for orb2 in range(bl_size):
                op = anticomm(comm(h_int,comm(h_int,c(bl,orb1))),c_dag(bl,orb2)) # Σ1 = {[H,[H,c]],c+} - ΣHF^2
                for term, coef in op:
                    
                    if len(term) == 2:
                        bl1, u1 = term[0][1]
                        bl2, u2 = term[1][1]
                        
                        sigma_1[bl][orb1,orb2] += coef*trace_rho_op(dm, c_dag(bl1,u1)*c(bl2,u2),hdiag)
                        
                    elif len(term) == 4:

                        bl1, u1 = term[0][1]
                        bl2, u2 = term[1][1]
                        bl3, u3 = term[2][1]
                        bl4, u4 = term[3][1]

                        sigma_1[bl][orb1,orb2] += coef*trace_rho_op(dm, c_dag(bl1,u1)*c_dag(bl2,u2)*c(bl3,u3)*c(bl4,u4),hdiag)
                
                sigma_1[bl][orb1,orb2] -= sigma_hf[bl][orb1,orb2]**2
    
    return [sigma_hf, sigma_1]

def repair_tail(Sigma, moments):
    
    Sigma_repair = Sigma.copy()
    
    iwn = np.array([x.real+1j*x.imag for x in Sigma_repair.mesh], dtype=complex)
    
    pos_freq = np.where(iwn.imag > 0)
    
    mid = len(iwn)//2
    
    for name, sig in Sigma_repair:
        bl_size = sig.target_shape[0]
        for orb1 in range(bl_size):
            for orb2 in range(bl_size):
                Sigma_asymp = moments[0][name][orb1,orb2]+moments[1][name][orb1,orb2]/iwn
                
                min_func = np.abs(sig.data[:,orb1,orb2].imag-Sigma_asymp.imag)
                replace = mid + np.argmin(min_func[pos_freq])
                mpi.report('Replaceing tail of Sigma_iw at omega = {}'.format(iwn[replace]))
                for i in range(replace,len(iwn)): Sigma_repair[name].data[i,orb1,orb2] = Sigma_asymp[i]
                for i in range(len(iwn)-replace): Sigma_repair[name].data[i,orb1,orb2] = Sigma_asymp[i]

    return Sigma_repair

# define parameters here ########################################################################
beta = 40.                # inverse temperature (1/eV)
U = 2.3                   # hubbard U parameter
J = 0.40                  # hubbard J parameter
nloops = 40               # number of DMFT loops needs 5-10 loops to converge
nk = 40                   # number of k points in each dimension
density_required = 4.     # target density for setting the chemical potential
n_orb = 3                 # number of orbitals
mu = 11.3715              # chemical potential
mix = 0.8                 # mixing factor
add_spin = False
w90_seedname = 'sr2ruo4'
w90_pathname = './'
n_iw = 1025

USE_DLR = False

outfile = 'sro_dmft_'
outfile = outfile+'dlr' if USE_DLR else outfile+'tail'

# solver
solver_params = {}
solver_params["random_seed"] = 123 * mpi.rank + 567
solver_params["length_cycle"] = 120
solver_params["n_warmup_cycles"] = int(1e4)
solver_params["n_cycles"] = int(4e7/mpi.size)
solver_params["imag_threshold"] = 1e-8
solver_params["off_diag_threshold"] = 1e-6

if USE_DLR:
    solver_params["perform_tail_fit"] = False
    solver_params["measure_density_matrix"] = True
    solver_params["use_norm_as_weight"] = True
else:
    solver_params["perform_tail_fit"] = True
    solver_params["fit_max_moment"] = 10
    solver_params["fit_min_w"] = 2
    solver_params["fit_max_w"] = 6

#################################################################################################


# initialize solver
S = Solver(beta=beta, gf_struct = [('up', 3), ('down', 3)], n_iw = n_iw, n_tau=10001, delta_interface=True, n_l = 31)

# set up interaction Hamiltonian
Umat, Upmat = util.U_matrix_kanamori(n_orb=n_orb, U_int=U, J_hund=J)
h_int = util.h_int_kanamori(['up', 'down'], range(n_orb), off_diag=True, U=Umat, Uprime=Upmat, J_hund=J)

Gloc = S.G_iw.copy() # local Green's function

# set up Wannier Hamiltonian
H_add_loc = np.zeros((n_orb, n_orb), dtype=complex)
H_add_loc += np.diag([-mu]*n_orb)

L = get_TBL(path=w90_pathname, name=w90_seedname, extend_to_spin=add_spin, add_local=H_add_loc)

# set up Sumk
sumk = SumkDiscreteFromLattice(lattice=L, n_points=nk)

# extract epsilon0 from hoppings and add
e0 = L.hoppings[(0, 0, 0)]
mpi.report('epsilon0 (impurity energies):\n',e0.real)


# begin the DMFT calculation

#check if there are previous runs in the outfile and if so restart from there

previous_iter = 0
mu = 0.
if mpi.is_master_node():
    ar = HDFArchive(outfile+'.h5', 'a')
    if 'iterations' in ar: # we have preivous iterations
        print('Restarting from previous iteration...', ar['iterations'])
        previous_iter = ar['iterations']
        S.Sigma_iw = ar['dmft_results/last_iter']['Sigma_iw']
        mu = ar['dmft_results/last_iter']['mu']
    else:
        print('Starting a new DMFT calculation...')
        ar.create_group('dmft_results')
        ar['dmft_results'].create_group('last_iter')
    del ar

    
previous_iter    = mpi.bcast(previous_iter)
S.Sigma_iw = mpi.bcast(S.Sigma_iw)
mu = mpi.bcast(mu)

for it in range(previous_iter,nloops+1):
    if mpi.is_master_node():
        print('-----------------------------------------------')
        print("Iteration = %s"%it)
        print('-----------------------------------------------')

    # determination of the next chemical potential via function Dens. Involves k summation
    mu, density = dichotomy(Density, mu, density_required, 1e-4, .5, max_loops = 100, x_name="chemical potential", y_name="density", verbosity=3)
    mpi.barrier()

    start_time = timer()
    Gloc << SK(mu = mu, Sigma = S.Sigma_iw)
    mpi.barrier()
    
    nlat = Gloc.total_density().real # lattice density
    if mpi.is_master_node():
        print('Gloc density matrix:')
        for block, gf in Gloc:
            print(block)
            print(gf.density().real)
            print('--------------')
        print('total occupation {:.4f}'.format(nlat))

    # calculate effective atomic levels (eal)
    solver_eal = e0 - np.diag([mu]*n_orb)
    Hloc_0 = Operator()
    for spin in ['up','down']:
        for o1 in range(n_orb):
            for o2 in range(n_orb):
                # check if off-diag element is larger than threshold
                if o1 != o2 and abs(solver_eal[o1,o2]) < solver_params['off_diag_threshold']:
                    continue
                else:
                    Hloc_0 += (solver_eal[o1,o2].real)/2 * (c_dag(spin,o1) * c(spin,o2) + c_dag(spin,o2) * c(spin,o1))

    solve_params['h_loc0'] = Hloc_0
    
    G0_iw = Gloc.copy()
    G0_iw << 0.0+0.0j
    G0_iw << inverse(S.Sigma_iw + inverse(Gloc))
    
    Delta_iw = G0_iw.copy()
    Delta_iw << 0.0+0.0j
    for name, g0 in G0_iw:
        Delta_iw[name] << iOmega_n - inverse(g0) - solver_eal
        known_moments = make_zero_tail(Delta_iw[name], 1)
        tail, err = fit_hermitian_tail(Delta_iw[name], known_moments)
        mpi.report('tail fit error Delta_iw for block {}: {}'.format(name,err))
        S.Delta_tau[name] << make_gf_from_fourier(Delta_iw[name], S.Delta_tau.mesh, tail).real
    
    if mpi.is_master_node():
        ar = HDFArchive(outfile+'.h5','a')

        ar['dmft_results']['iterations'] = it
        ar['dmft_results'].create_group('it_{}'.format(it))

        ar['dmft_results/it_{}'.format(it)]['G0_iw'] = G0_iw
        ar['dmft_results/last_iter']['G0_iw'] = G0_iw
        
        ar['dmft_results/it_{}'.format(it)]['Delta_tau'] = S.Delta_tau
        ar['dmft_results/last_iter']['Delta_tau'] = S.Delta_tau
        
        ar['dmft_results/it_{}'.format(it)]['Gloc'] = Gloc
        ar['dmft_results/last_iter']['Gloc'] = Gloc
        
        ar['dmft_results/it_{}'.format(it)]['n_latt'] = nlat
        ar['dmft_results/last_iter']['n_latt'] = nlat
        
        ar['dmft_results/it_{}'.format(it)]['mu'] = mu
        ar['dmft_results/last_iter']['mu'] = mu

        del ar
    
    # solve the impurity problem. The solver is performing the dyson equation as postprocessing
    S.solve(h_int=h_int, **solver_params)

    if USE_DLR:
        G_iw = fit_dlr(S.G_iw, lamb=dlr_lambda)

        Sigma_iw_dlr = inverse(G0_iw) - inverse(G_iw)

        analytic_moments = calculate_analytic_moments(S.density_matrix,
                                                      S.h_loc_diagonalization,
                                                      S.gf_struct,
                                                      S.last_solve_parameters['h_int']
                                                      )
        if mpi.is_master_node():
            print("Calculated analytic moments of the self-energy")
            for moment in analytic_moments:
                for key in moment.keys():
                    print(key)
                    print(moment[key])

        Sigma_iw = repair_tail(Sigma_iw_dlr, analytic_moments)

        S.Sigma_iw << Sigma_iw
        S.G_iw << G_iw

    
    G_iw = S.G_iw.copy()
    nimp = G_iw.total_density().real #impurity density
    if mpi.is_master_node():
        print('impurity density matrix:')
        for block, gf in G_iw:
            print(block)
            print(gf.density().real)
            print('--------------')
        print('total occupation {:.4f}'.format(nimp))
    mpi.report('Impurity density is {:.4f}'.format(nimp))


    # force self energy obtained from solver to be hermitian
    for name, s_iw in S.Sigma_iw:
        S.Sigma_iw[name] = make_hermitian(s_iw)
    
    # symmetrize the self-energy                            
    S.Sigma_iw['up'] << .5*(S.Sigma_iw['up'] + S.Sigma_iw['down'])
    S.Sigma_iw['down'] << S.Sigma_iw['up']
      
    if mpi.is_master_node():
        ar = HDFArchive(outfile+'.h5','a')
        ar['dmft_results/it_{}'.format(it)]['G_tau'] = S.G_tau
        ar['dmft_results/last_iter']['G_tau'] = S.G_tau
        
        ar['dmft_results/it_{}'.format(it)]['G_iw'] = S.G_iw
        ar['dmft_results/last_iter']['G_iw'] = S.G_iw
        
        ar['dmft_results/it_{}'.format(it)]['Sigma_iw'] = S.Sigma_iw
        ar['dmft_results/last_iter']['Sigma_iw'] = S.Sigma_iw
        
        ar['dmft_results/it_{}'.format(it)]['n_imp'] = nimp
        ar['dmft_results/last_iter']['n_imp'] = nimp

        if USE_DLR:
            ar['dmft_results/it_{}'.format(it)]['moments_self_energy'] = analytic_moments
            ar['dmft_results/last_iter']['moments_self_energy'] =  analytic_moments

            ar['dmft_results/it_{}'.format(it)]['density_matrix'] = density_matrix
            ar['dmft_results/last_iter']['density_matrix'] =  density_matrix

            ar['dmft_results/it_{}'.format(it)]['h_loc_diag'] = h_loc_diagonalization
            ar['dmft_results/last_iter']['h_loc_diag'] =  h_loc_diagonalization


        if it > 1:
            S.Sigma_iw << mix * S.Sigma_iw + (1.0-mix) * ar['dmft_results/it_{}'.format(it-1)]['Sigma_iw']
        del ar
        
    S.Sigma_iw << mpi.bcast(S.Sigma_iw)
