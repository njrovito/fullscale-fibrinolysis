import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #
# FLATiron
from flatiron_tk.functions import IndicatorFieldScalar
from flatiron_tk.io import h5_mod
from flatiron_tk.io import InputObject
from flatiron_tk.mesh import Mesh

from flatiron_tk.physics import TransientMultiPhysicsProblem
from flatiron_tk.solver import PhysicsSolver
import fenics as fe
# ------------------------------------------------------- #
# Block solver
from petsc4py import PETSc
from collections.abc import Iterable
# We can use MPI that's wrapped aroung the Mesh object, but 
# this is a little more straight-forward.
from mpi4py import MPI
from flatiron_tk.solver import BlockNonLinearSolver
from flatiron_tk.solver import ConvergenceMonitor
from flatiron_tk.solver import FieldSplitTree
from flatiron_tk.solver import NonLinearProblem
# ------------------------------------------------------- #
# Fibrinolysis model
import tools.fibrinolysis_solverLib as fibsol
# ------------------------------------------------------- #

from tools.brinkman_nse import BrinkmanNavierStokes
from tools.fibrinolysis_cfd_tools import FibrinolysisSetup
from tools.fibrinolysis_cfd_tools import FieldScalar

from tools.fibrinolysis_cfd_tools import set_apl_transport
from tools.fibrinolysis_cfd_tools import set_fbg_transport
from tools.fibrinolysis_cfd_tools import set_pls_transport
from tools.fibrinolysis_cfd_tools import set_plg_transport
from tools.fibrinolysis_cfd_tools import set_tPA_transport
from tools.fibrinolysis_cfd_tools import get_inlet_areas
from tools.fibrinolysis_cfd_tools import get_inlet_flow_ratios
from tools.fibrinolysis_cfd_tools import get_inlet_flowrate
from tools.fibrinolysis_cfd_tools import get_inlet_velocities
from tools.fibrinolysis_cfd_tools import get_normal_vector
from tools.fibrinolysis_cfd_tools import get_tpa_flux

class MyNonLinearSolver(fe.NewtonSolver):
	def __init__(self, comm, problem, la_solver, **kwargs):
		self.problem = problem
		self.solver_type = kwargs.pop('solver_type', 'gmres')
		self.pc_type     = kwargs.pop('pc_type'    , 'hypre')
		self.rel_tol     = kwargs.pop('relative_tolerance', 1e-8)
		self.abs_tol     = kwargs.pop('absolute_tolerance', 1e-10)
		self.max_iter    = kwargs.pop('maximum_iterations', 1000)
		fe.NewtonSolver.__init__(self, comm,
		fe.PETScKrylovSolver(), fe.PETScFactory.instance())

	def solver_setup(self, A, P, problem, iteration):
		self.linear_solver().set_operator(A)
		fe.PETScOptions.set("ksp_type", self.solver_type)
		fe.PETScOptions.set("ksp_monitor")
		fe.PETScOptions.set("pc_type", self.pc_type)
		self.linear_solver().parameters["relative_tolerance"] = self.rel_tol
		self.linear_solver().parameters["absolute_tolerance"] = self.abs_tol
		self.linear_solver().parameters["maximum_iterations"] = self.max_iter
		self.linear_solver().set_from_options()

	def solve(self):
		sol_vector = self.problem.physics.solution
		super().solve(self.problem, sol_vector.vector())

class MyPhysicsSolver(PhysicsSolver):
	def set_problem_solver(self):
		self.problem_solver = MyNonLinearSolver(self.physics.mesh.comm, self.problem, self.la_solver)

def build_nse_block_solver(nse):
	'''
	New block Krylov solver.
	'''
	problem = NonLinearProblem(nse)
	def set_ksp_u(ksp):
		ksp.setType(PETSc.KSP.Type.FGMRES)
		ksp.setMonitor(ConvergenceMonitor("KSP_u", verbose=False))
		ksp.setTolerances(max_it=10)
		ksp.pc.setType(PETSc.PC.Type.JACOBI)
		ksp.setUp()

	def set_ksp_p(ksp):
		ksp.setType(PETSc.KSP.Type.FGMRES)
		ksp.setMonitor(ConvergenceMonitor("KSP_p", verbose=False))
		ksp.setTolerances(max_it=3)
		ksp.pc.setType(PETSc.PC.Type.HYPRE)
		ksp.pc.setHYPREType("boomeramg")
		ksp.setUp()

	def set_outer_ksp(ksp):
		ksp.setType(PETSc.KSP.Type.FGMRES)
		ksp.setGMRESRestart(50)
		ksp.setTolerances(rtol=1e-100, atol=1e-10)
		ksp.setMonitor(ConvergenceMonitor("KSP"))

	split = {'fields': ('u', 'p'),
				'composite_type': 'schur',
				'schur_fact_type': 'full',
				'schur_pre_type': 'a11',
				'ksp0_set_function': set_ksp_u,
				'ksp1_set_function': set_ksp_p}
	tree = FieldSplitTree(nse, split)
	solver = BlockNonLinearSolver(tree, nse.mesh.comm, problem, fe.PETScKrylovSolver())
	return solver

def main(path_to_vessels, path_to_thrombus, path_to_fibrin_constants='fibrinolysis_inputs.xml'):
	output_dir = 'output-409-m'
	output_type = 'h5'
	debug_outputs = False
	fsu = FibrinolysisSetup(path_fibrin_constants)
	CONSTANTS = fsu.constants
	theta = 0.5

	t = 0
	count = 0
	nse_time = 0
	stp_time = 0
	rxn_time = 0
	
	# Constant quantities
	dt = 0.002 # 002  # s
	end_time = 1.0  # s
	output_period = 20
	zero_vector = fe.Constant((0.0, 0.0, 0.0))
	zero_scalar = fe.Constant(0.0)

	# Boundary ids
	id_LMCA = 2
	id_RPCA = 3
	id_RMCA = 4
	id_LPCA = 5
	id_LICA = 6
	id_RICA = 7
	id_LACA = 8
	id_wall = 9
	id_RACA = 10
	id_BA = 11

	inout_boundary_list = [
					
					]
	outlet_boundary_list = [id_LMCA, id_RPCA, id_RMCA, id_LPCA, id_LACA, id_RACA]

	# initialize fibrin concentrations
	n_tot_init = fsu.n_tot_init	

	n_0 = np.array([n_tot_init, CONSTANTS['init_tPA_bindSites'], CONSTANTS['init_Plasminogen_bindSites'],
					CONSTANTS['init_Plasmin_bindSites'], CONSTANTS['init_Plasmin_bindConc']]).T
	# endregion

	# --------------------------------- #
	# region: Read mesh(es)
	# --------------------------------- #
	mesh_file = path_to_vessels
	mesh = Mesh(mesh_file=mesh_file)
	
	thrombus_file = path_to_thrombus
	thrombus_mesh = Mesh(mesh_file=thrombus_file, comm=fe.MPI.comm_self) 

	rank = mesh.comm.rank
	size = mesh.comm.size
	if debug_outputs: print(size, rank)
	
	def inside_thrombus(x):
		return thrombus_mesh.mesh.bounding_box_tree().compute_first_entity_collision(fe.Point(x)) \
			< thrombus_mesh.mesh.num_cells()

	V = fe.FunctionSpace(mesh.mesh, 'CG', 1)  # for creating all custome mesh functions

	# define a vector that sets cells in/out of the domain to 1 and interior/wall cells to 0
	cell_type = fe.Function(V)
	cell_type.vector()[:] = 0
	for bnd in inout_boundary_list:
		bc = fe.DirichletBC(V, fe.Constant(1), mesh.boundary, bnd)
		bc.apply(cell_type.vector())
	
	cell_type_array = cell_type.vector().gather_on_zero()
	if rank == 0:
		cell_type_array = np.array(cell_type_array)
	
	cell_type_array = mesh.comm.bcast(cell_type_array, root=0)
		
	# Define Navier-Stokes problem
	nse = BrinkmanNavierStokes(mesh)
	nse.set_element('CG', 1, 'CG', 1)
	nse.build_function_space()
	nse.set_time_step_size(dt)
	nse.set_mid_point_theta(theta)
	nse.set_density(fsu.density)
	nse.set_dynamic_viscosity(fsu.viscosity)
	nse.set_outlet_ids(outlet_boundary_list)
	
	# Define thrombus domain
	thrombus_identifier = FieldScalar(inside_thrombus)
	thrombus_identifier.set_inside_value(1)
	thrombus_identifier.set_outside_value(0)
	thrombus_domain = fe.interpolate(thrombus_identifier, nse.V.sub(1).collapse())
	
	phi_output_file = output_dir + '/phi.' + output_type
	if rank == 0: print('Output file set to {}'.format(phi_output_file))

	if output_type == 'pvd':
		phi_output_file = fe.File(phi_output_file)

	if output_type == 'h5':
		phi_output_file = phi_output_file
		phi_output_fid = h5_mod.h5_init_output_file(phi_output_file, mesh=mesh.mesh, boundaries=mesh.boundary)

	thrombus_identifier.set_inside_value(fsu.init_porosity)
	thrombus_identifier.set_outside_value(1.0)
	phi_0 = fe.interpolate(thrombus_identifier, V)
	phi = phi_0
	nse.set_partical_char_len(fsu.particle_char_length)
	nse.set_porosity(phi)
	nse.set_weak_form(thrombus_domain, stab=True)
	nse.set_writer(output_dir, output_type)

	# Define ADR problems for each species
	D = fsu.diffusivity_outside_thrombus - thrombus_domain * (fsu.diffusivity_outside_thrombus - fsu.diffusivity_inside_thrombus)  # define diffusivity by unit step function 
	(u0, _) = fe.split(nse.previous_solution)
	(un, _) = fe.split(nse.solution)

	# Set all scalar transport sub-classes (defined above)
	tpa_adr = set_tPA_transport(mesh, dt, theta, u0, un, D)
	plg_adr = set_plg_transport(mesh, dt, theta, u0, un, D)
	pls_adr = set_pls_transport(mesh, dt, theta, u0, un, D)
	fbg_adr = set_fbg_transport(mesh, dt, theta, u0, un, D)
	apl_adr = set_apl_transport(mesh, dt, theta, u0, un, D)
	
	# Define coupled NSE-ADR problem
	coupled_adr = TransientMultiPhysicsProblem(tpa_adr, plg_adr, pls_adr, fbg_adr, apl_adr)
	coupled_adr.set_element()
	coupled_adr.build_function_space()

	# Set solutions functions
	tpa = coupled_adr.solution_function('C_tpa')
	plg = coupled_adr.solution_function('C_plg')
	pls = coupled_adr.solution_function('C_pls')
	fbg = coupled_adr.solution_function('C_fbg')
	apl = coupled_adr.solution_function('C_apl')

	# --------------------------------- #
	# region: Set weak form
	# --------------------------------- #
	stp_options = {'stab':True}
	coupled_adr.set_weak_form(stp_options,stp_options,stp_options,stp_options,stp_options)
	coupled_adr.set_writer(output_dir, output_type)

	# endregion

	# region: create bind site mesh functions (n)
	n_tot = fe.Function(V)
	n_tpa = fe.Function(V)
	n_plg = fe.Function(V)
	n_pls = fe.Function(V)
	L_pls = fe.Function(V)
	n_tot.vector()[:] = n_0[0]
	n_tpa.vector()[:] = n_0[1]
	n_plg.vector()[:] = n_0[2]
	n_pls.vector()[:] = n_0[3]
	L_pls.vector()[:] = n_0[4]
	# endregion

	# --------------------------------- #
	# region: Set boundary conditions
	# --------------------------------- #
	# Get inward normal vectors
	BA_normal = -get_normal_vector(mesh.mesh, mesh.dim, nse.ds(id_BA))
	LICA_normal = -get_normal_vector(mesh.mesh, mesh.dim, nse.ds(id_LICA))
	RICA_normal = -get_normal_vector(mesh.mesh, mesh.dim, nse.ds(id_RICA))
	# Get velocity magnitudes at each inlet (t=0)	
	Q_max = fsu.total_max_inlet_flowrate 
	areas = get_inlet_areas(nse, [id_BA, id_LICA, id_RICA])
	ratios = get_inlet_flow_ratios(fsu)
	u_BA, u_LICA, u_RICA = get_inlet_velocities(Q_max, t, ratios, areas)

	# Plug inlet profiles into fenics expressions that can be updated for pulse flow
	inlet_BA_u = fe.Expression(('u * nx', 'u * ny', 'u * nz'), degree=1, u=u_BA, nx=BA_normal[0], ny=BA_normal[1], nz=BA_normal[2])
	inlet_LICA_u = fe.Expression(('u * nx', 'u * ny', 'u * nz'), degree=1, u=u_LICA, nx=LICA_normal[0], ny=LICA_normal[1], nz=LICA_normal[2])
	inlet_RICA_u = fe.Expression(('u * nx', 'u * ny', 'u * nz'), degree=1, u=u_RICA, nx=RICA_normal[0], ny=RICA_normal[1], nz=RICA_normal[2])
	# Get total flux through each inlet of tPA
	tpa_flux_magntitude = fsu.total_flux_tpa
	q_BA_tpa, q_LICA_tpa, q_RICA_tpa = get_tpa_flux(tpa_flux_magntitude, areas)
	flux_BA_tpa = fe.Expression(('q * nx', 'q * ny', 'q * nz'), degree=1, q=q_BA_tpa, nx=-BA_normal[0], ny=-BA_normal[1], nz=-BA_normal[2])
	flux_LICA_tpa = fe.Expression(('q * nx', 'q * ny', 'q * nz'), degree=1, q=q_LICA_tpa, nx=-LICA_normal[0], ny=-LICA_normal[1], nz=-LICA_normal[2])
	flux_RICA_tpa = fe.Expression(('q * nx', 'q * ny', 'q * nz'), degree=1, q=q_RICA_tpa, nx=-RICA_normal[0], ny=-RICA_normal[1], nz=-RICA_normal[2])
	C_tpa_0 = coupled_adr.sub_physics[0].previous_solution
	C_tpa_n = coupled_adr.sub_physics[0].solution

	u_bcs = {
			id_BA  : {'type': 'dirichlet', 'value': inlet_BA_u},
			id_LICA : {'type': 'dirichlet', 'value': inlet_LICA_u},
			id_RICA : {'type': 'dirichlet', 'value': inlet_RICA_u},
			id_wall : {'type': 'dirichlet', 'value': zero_vector}
			}
	p_bcs = {
			id_LACA : {'type': 'dirichlet', 'value': zero_scalar},
			id_RACA : {'type': 'dirichlet', 'value': zero_scalar},
			id_LPCA : {'type': 'dirichlet', 'value': zero_scalar},
			id_RPCA : {'type': 'dirichlet', 'value': zero_scalar},
			id_LMCA : {'type': 'dirichlet', 'value': zero_scalar},
			id_RMCA : {'type': 'dirichlet', 'value': zero_scalar}
			}
	tpa_bcs = {
			id_wall : {'type': 'dirichlet', 'value': fe.Constant(0)},
			id_BA   : {'type': 'neumann', 'value': 0.5 * (2 * flux_BA_tpa + un * C_tpa_n + u0 * C_tpa_0)},
			id_LICA : {'type': 'neumann', 'value': 0.5 * (2 * flux_LICA_tpa + un * C_tpa_n + u0 * C_tpa_0)},
			id_RICA : {'type': 'neumann', 'value': 0.5 * (2 * flux_RICA_tpa + un * C_tpa_n + u0 * C_tpa_0)}
			}
	
	plg_bcs = {	
			id_wall : {'type': 'dirichlet', 'value': fe.Constant(0)},
			id_BA   : {'type': 'dirichlet', 'value': fe.Constant(fsu.plg_blood_concentration_init)},
			id_LICA : {'type': 'dirichlet', 'value': fe.Constant(fsu.plg_blood_concentration_init)},
			id_RICA : {'type': 'dirichlet', 'value': fe.Constant(fsu.plg_blood_concentration_init)}
			}
	pls_bcs = {
			id_wall : {'type': 'dirichlet', 'value': fe.Constant(0)},
			id_BA   : {'type': 'dirichlet', 'value': fe.Constant(fsu.pls_blood_concentration_init)},
			id_LICA : {'type': 'dirichlet', 'value': fe.Constant(fsu.pls_blood_concentration_init)},
			id_RICA : {'type': 'dirichlet', 'value': fe.Constant(fsu.pls_blood_concentration_init)}
			}
	fbg_bcs = {
			id_wall : {'type': 'dirichlet', 'value': fe.Constant(0)},
			id_BA   : {'type': 'dirichlet', 'value': fe.Constant(fsu.fbg_blood_concentration_init)},
			id_LICA : {'type': 'dirichlet', 'value': fe.Constant(fsu.fbg_blood_concentration_init)},
			id_RICA : {'type': 'dirichlet', 'value': fe.Constant(fsu.fbg_blood_concentration_init)}
			}
	apl_bcs = {
			id_wall : {'type': 'dirichlet', 'value': fe.Constant(0)},
			id_BA   : {'type': 'dirichlet', 'value': fe.Constant(fsu.apl_blood_concentration_init)},
			id_LICA : {'type': 'dirichlet', 'value': fe.Constant(fsu.apl_blood_concentration_init)},
			id_RICA : {'type': 'dirichlet', 'value': fe.Constant(fsu.apl_blood_concentration_init)}
			}

	nse_bc_dict = {
				'u': u_bcs,
				'p': p_bcs,
				}
	nse.set_bcs(nse_bc_dict)
	stp_bc_dict = {
				'C_tpa': tpa_bcs,
				'C_plg': plg_bcs,
				'C_pls': pls_bcs,
				'C_fbg': fbg_bcs,
				'C_apl': apl_bcs
				}
	coupled_adr.set_bcs(stp_bc_dict)
	
	stp_la_solver = fe.PETScKrylovSolver()
	stp_solver = PhysicsSolver(coupled_adr, stp_la_solver)
	nse_solver = build_nse_block_solver(nse)
	# endregion

	while t < end_time:
		t += dt
		count += 1

		# Update velocity inlets	
		u_BA, u_LICA, u_RICA = get_inlet_velocities(Q_max, t, ratios, areas)
		inlet_BA_u.u = u_BA
		inlet_LICA_u.u = u_LICA
		inlet_RICA_u.u = u_RICA

		nse_solver.solve()
		
		nse.update_previous_solution()
		
		
		stp_start = time.time()

		# (u0, _) = fe.split(nse.previous_solution)
		# (un, _) = fe.split(nse.solution)
		# tpa_adr.set_advection_velocity(u0, un)
		# plg_adr.set_advection_velocity(u0, un)
		# pls_adr.set_advection_velocity(u0, un)
		# fbg_adr.set_advection_velocity(u0, un)
		# apl_adr.set_advection_velocity(u0, un)
		# stp_solver.solve()
		# coupled_adr.update_previous_solution()

		# stp_stop = time.time()
		# stp_time += (stp_stop - stp_start)
		

		# # --------------------------------- #
		# # region: Solve reaction substep
		# # --------------------------------- #
		# rxn_start = time.time()
		# c_tpa, c_plg, c_pls, c_fbg, c_apl = coupled_adr.solution_function().split(deepcopy=True)

		# # region: Get current solution vectors
		# tpa_array = c_tpa.vector().get_local()
		# plg_array = c_plg.vector().get_local()
		# pls_array = c_pls.vector().get_local()
		# fbg_array = c_fbg.vector().get_local()
		# apl_array = c_apl.vector().get_local()
		# n_tot_array = n_tot.vector().get_local()
		# n_tpa_array = n_tpa.vector().get_local()
		# n_plg_array = n_plg.vector().get_local()
		# n_pls_array = n_pls.vector().get_local()
		# L_pls_array = L_pls.vector().get_local()

		# phi_array = phi.vector().get_local()
		# phi_0_array = phi_0.vector().get_local()
		# # endregion
		
		# count = 0
		# test = 0

		# for i in range(len(phi_array)):			
		# 	if cell_type_array[i] == 0:
		# 		test += 1
		# 		RK4_solution = fibsol.compModelSolver_RK4(CONSTANTS, t, dt, tpa_array[i], plg_array[i], pls_array[i], fbg_array[i], apl_array[i],
		# 		n_tot_array[i], n_tpa_array[i], n_plg_array[i], n_pls_array[i], L_pls_array[i],
		# 		solveN=True, solveR=True)

		# 		tpa_array[i] = RK4_solution[0]
		# 		plg_array[i] = RK4_solution[1]
		# 		pls_array[i] = RK4_solution[2]
		# 		fbg_array[i] = RK4_solution[3]
		# 		apl_array[i] = RK4_solution[4]
		# 		n_tot_array[i] = RK4_solution[5]
		# 		n_tpa_array[i] = RK4_solution[6]
		# 		n_plg_array[i] = RK4_solution[7]
		# 		n_pls_array[i] = RK4_solution[8]
		# 		L_pls_array[i] = RK4_solution[9]
				
		# 		if phi_array[i] <= 0.99 and phi_0_array[i] != 1:  # Prevents porosity calculation outside thrombus region (results in 1)
		# 			phi_array[i] = 1 - (1 - phi_0_array[i]) * (n_tot_array[i] / n_tot_init)	


		# # region: Set concentration and bind site functions based on compartmental model	
		# c_tpa.vector().set_local(tpa_array)
		# c_tpa.vector().apply('') 
		# c_plg.vector().set_local(plg_array)
		# c_plg.vector().apply('')  
		# c_pls.vector().set_local(pls_array)
		# c_pls.vector().apply('')  
		# c_fbg.vector().set_local(fbg_array)
		# c_fbg.vector().apply('')  
		# c_apl.vector().set_local(apl_array)
		# c_apl.vector().apply('') 

		# # These can technically be numpy arrays only since they are not output as mesh functions
		# n_tot.vector().set_local(n_tot_array)
		# n_tot.vector().apply('')
		# n_tpa.vector().set_local(n_tpa_array)
		# n_tpa.vector().apply('')
		# n_plg.vector().set_local(n_plg_array)
		# n_plg.vector().apply('')
		# n_pls.vector().set_local(n_pls_array)
		# n_pls.vector().apply('')
		# L_pls.vector().set_local(L_pls_array)
		# L_pls.vector().apply('')
		# # endregion
		
		# # phi.vector().set_local(phi_array)
		# porosity_field = fe.project(phi, V)		
		# # endregion

		# rxn_stop = time.time()
		# rxn_time += (rxn_stop - rxn_start)

		# if count % output_period == 0:  # Save solutions at time step
		# 	porosity_field.rename('phi', 'phi')

		# 	if output_type == 'pvd':  # Save phi as pvd
		# 		phi_output_file.write(porosity_field, t)
		# 		if rank == 0: print('output phi.pvd')
			
		# 	elif output_type == 'h5':  # Save phi as h5
		# 		h5_mod.h5_write(porosity_field, 'phi', h5_object=phi_output_fid, timestamp=t)
		# 		if rank == 0: print('output phi.h5')

			# nse.write(time_stamp=t)
			# coupled_adr.write(time_stamp=t)

		if rank == 0:  # Print current time step solved 
			print('-'*50)
			print('solved time: {}'.format(t))
			print('-'*50)

	return

if __name__ == '__main__':
	path_vessels = 'mesh-scrap/mesh-3d/patient409/RMCA-fine/test.h5'
	path_thrombus = 'RMCA-mid/occlusion-13-0473.h5'
	path_fibrin_constants = 'fibrinolysis_inputs.xml'

	main(path_vessels, path_thrombus, path_fibrin_constants)