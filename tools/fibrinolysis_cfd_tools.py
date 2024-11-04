import numpy as np
import fenics as fe
import tools.fibrinolysis_initializationLib as fibinit
import tools.fibrinolysis_reactionLib as fibrxn
import tools.fibrinolysis_solverLib as fibsol
from flatiron_tk.physics import TransientScalarTransport

class FibrinolysisSetup():
	def __init__(self, path_to_inputs) -> None:
		self.constants = fibinit.readInputs(path_to_inputs)

		self.initialize_bind_site()
		self.parse_blood_parameters()
		self.parse_thrombus_parameters()
		self.parse_flow_parameters()
		self.parse_patient_parameters()

	def initialize_bind_site(self):
		self.n_tot_init = fibinit.initFibrinSites(self.constants)

	def parse_blood_parameters(self):
		self.density = self.constants['Blood_density']
		self.viscosity = self.constants['Blood_viscosity']
	
	def parse_thrombus_parameters(self):
		self.init_porosity = self.constants['Initial_thrombus_porosity']
		self.particle_char_length = self.constants['Particle_characteristic_length']
		self.diffusivity_inside_thrombus = self.constants['Diffusivity_inside_thrombus']
		self.diffusivity_outside_thrombus = self.constants['Diffusivity_outside_thrombus']

	def parse_flow_parameters(self):
		self.total_max_inlet_flowrate = self.constants['Total_max_inlet_flowrate']
		self.total_max_BA_flowrate = self.constants['Total_max_BA_flowrate']
		self.total_max_LICA_flowrate = self.constants['Total_max_LICA_flowrate']
		self.total_max_RICA_flowrate = self.constants['Total_max_RICA_flowrate']

		self.BA_flow_ratio = self.total_max_BA_flowrate / self.total_max_inlet_flowrate
		self.LICA_flow_ratio = self.total_max_LICA_flowrate / self.total_max_LICA_flowrate
		self.RICA_flow_ratio = self.total_max_RICA_flowrate / self.total_max_RICA_flowrate

		self.total_max_inlet_flowrate /= 60

	def parse_patient_parameters(self):
		self.patient_mass = self.constants['patient_mass']  									# [kg]
		self.treatment_dosage = self.constants['treatment_dosage']  							# [mg_tpa/kg_mass]
		self.treatment_time = self.constants['infusion_time']									# [s]
		self.tpa_blood_concentration_itit = self.constants['init_tPA_conc'] / 1e6 				# [uMol/mm^3]
		self.plg_blood_concentration_init = self.constants['init_Plasminogen_conc'] / 1e6   	# [uMol/mm^3]
		self.pls_blood_concentration_init = self.constants['init_Plasmin_conc']  / 1e6      	# [uMol/mm^3]
		self.fbg_blood_concentration_init = self.constants['init_Fibrinogen_conc']  / 1e6   	# [uMol/mm^3]
		self.apl_blood_concentration_init = self.constants['init_Antiplasmin_conc']  / 1e6  	# [uMol/mm^3]
		self.molecular_weight_tpa = self.constants['Molecular_weight_tPA']						# [Dalton]
		self.total_flux_tpa = self.treatment_dosage * self.patient_mass / self.treatment_time / 1000 / self.molecular_weight_tpa * 1e6  # [uMol/s]

class FieldScalar(fe.UserExpression):
	'''
	Define a heaviside-type field where
	I = self.inside_value if indicator_domain(x):
	and I = self.outside_value otherwise
	'''
	def __init__(self, indicator_domain, **kwargs):
		self.indicator_domain = indicator_domain
		super().__init__(**kwargs)

	def eval(self, value, x):
		if self.indicator_domain(x):
			value[0] = self.inside_value
		else:
			value[0] = self.outside_value
		return value

	def set_inside_value(self, inside_value):
		self.inside_value = inside_value

	def set_outside_value(self, outside_value):
		self.outside_value = outside_value

	def value_shape(self):
		return ()

def set_tPA_transport(mesh, dt, theta, u0, un, D):
	# -------------------------------
	# Define tPA advection-diffusion
	# -------------------------------
	tpa_adr = TransientScalarTransport(mesh, dt, theta=theta)
	tpa_adr.set_tag('C_tpa')
	tpa_adr.set_element('CG', 1)
	tpa_adr.build_function_space()
	tpa_adr.set_advection_velocity(u0, un)
	tpa_adr.set_diffusivity(D, D)
	tpa_adr.set_reaction(0, 0)
	# reactions handled later; set_reaction adds reaction term to
	# time-discretized theta-galerkin, we will define reactions explicitly
	return tpa_adr

def set_plg_transport(mesh, dt, theta, u0, un, D):
	# ---------------------------------------
	# Define plasminogen advection-diffusion
	# ---------------------------------------
	plg_adr = TransientScalarTransport(mesh, dt, theta=theta)
	plg_adr.set_tag('C_plg')
	plg_adr.set_element('CG', 1)
	plg_adr.build_function_space()
	plg_adr.set_advection_velocity(u0, un)
	plg_adr.set_diffusivity(D, D)
	plg_adr.set_reaction(0, 0)
	# reactions handled later; set_reaction adds reaction term to
	# time-discretized theta-galerkin, we will define reactions explicitly
	return plg_adr

def set_pls_transport(mesh, dt, theta, u0, un, D):
	# ---------------------------------------
	# Define plasmin advection-diffusion
	# ---------------------------------------
	pls_adr = TransientScalarTransport(mesh, dt, theta=theta)
	pls_adr.set_tag('C_pls')
	pls_adr.set_element('CG', 1)
	pls_adr.build_function_space()
	pls_adr.set_advection_velocity(u0, un)
	pls_adr.set_diffusivity(D, D)
	pls_adr.set_reaction(0, 0)
	# reactions handled later; set_reaction adds reaction term to
	# time-discretized theta-galerkin, we will define reactions explicitly	
	return pls_adr
	
def set_fbg_transport(mesh, dt, theta, u0, un, D):
	# ---------------------------------------
	# Define fibrinogen advection-diffusion
	# ---------------------------------------
	fbg_adr = TransientScalarTransport(mesh, dt, theta=theta)
	fbg_adr.set_tag('C_fbg')
	fbg_adr.set_element('CG', 1)
	fbg_adr.build_function_space()
	fbg_adr.set_advection_velocity(u0, un)
	fbg_adr.set_diffusivity(D, D)
	fbg_adr.set_reaction(0, 0)
	# reactions handled later; set_reaction adds reaction term to
	# time-discretized theta-galerkin, we will define reactions explicitly
	return fbg_adr

def set_apl_transport(mesh, dt, theta, u0, un, D):
	# ---------------------------------------
	# Define antiplasmin advection-diffusion
	# ---------------------------------------
	apl_adr = TransientScalarTransport(mesh, dt, theta=theta)
	apl_adr.set_tag('C_apl')
	apl_adr.set_element('CG', 1)
	apl_adr.build_function_space()
	apl_adr.set_advection_velocity(u0, un)
	apl_adr.set_diffusivity(D, D)
	apl_adr.set_reaction(0, 0)
	# reactions handled later; set_reaction adds reaction term to
	# time-discretized theta-galerkin, we will define reactions explicitly
	return apl_adr

def get_normal_vector(mesh, dim, boundary):
	# mesh = mesh.mesh; dim = mesh.dim; boundary = physics.dy(boundary_id)
	n = fe.FacetNormal(mesh)
	normal = np.array([fe.assemble(n[i] * boundary) for i in range(dim)])
	normal_mag = np.linalg.norm(normal, 2)
	normal_vector = (1/normal_mag) * normal
	return normal_vector

def get_inlet_areas(nse_class, ids):
	# one is a fenics "domain" so we can use fe.assemble to integrate
	# Literally int_{Gamma_i}(1 ds_Gamma_i)
	one = fe.interpolate(fe.Constant(1), nse_class.V.sub(1).collapse())
	area_BA = fe.assemble(one * nse_class.ds(ids[0]))
	area_LICA = fe.assemble(one * nse_class.ds(ids[1]))
	area_RICA = fe.assemble(one * nse_class.ds(ids[2]))
	return [area_BA, area_LICA, area_RICA]

def get_inlet_flowrate(Q_max, t):
	cardiac_cycle_period = 0.9
	systolic_time = cardiac_cycle_period / 3
	# Define peak of the model inlet flowrate as a fraction of the peak of the sin pulse model
	alpha = 0.8
	# Define the period of the sin term (a function of alpha, found with Excel solver)
	T = 0.3099996187977
	# Define time relative to cycle (tau is time within cycle)
	tau = round(t % cardiac_cycle_period, 3)
	# Solve for total inlet flowrate
	if tau < systolic_time:
		flowrate = alpha * Q_max * np.sin(np.pi * tau / T)
	else:
		flowrate = -alpha * Q_max * np.sin(systolic_time * np.pi / T) / (cardiac_cycle_period - T) * (tau - T) + alpha * Q_max * np.sin(systolic_time * np.pi / T)
	return flowrate

def get_inlet_velocities(Q_max, t, ratios, areas):
	flowrate = get_inlet_flowrate(Q_max, t)
	# Compute the velocity as the fraction of flow through each vessel
	BA_vel = flowrate * ratios[0] / areas[0]
	LICA_vel = flowrate * ratios[1] / areas[1]
	RICA_vel = flowrate * ratios[2] / areas[2]
	return BA_vel, LICA_vel, RICA_vel

def get_inlet_flow_ratios(setup_class):
	return [setup_class.BA_flow_ratio, setup_class.LICA_flow_ratio, setup_class.RICA_flow_ratio]

def get_tpa_flux(total_flux, areas):
	flux_BA = total_flux / areas[0]
	flux_LICA = total_flux / areas[1]
	flux_RICA = total_flux / areas[2]
	return flux_BA, flux_LICA, flux_RICA