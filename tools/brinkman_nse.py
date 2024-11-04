import fenics as fe

from flatiron_tk.physics import IncompressibleNavierStokes

class BrinkmanNavierStokes(IncompressibleNavierStokes):
	def set_porosity(self, porosity):
		self.porosity = porosity 
	
	def set_outlet_ids(self, list):
		self.outlet_id_list = list

	def set_partical_char_len(self, char_len):
		self.char_len = fe.Constant(char_len)

	def set_weak_form(self, I, stab=False):
		# super.set_weak_form()
		mu = self.external_function('dynamic viscosity')
		rho = self.external_function('density')
		dt = self.external_function('dt')
		theta = self.external_function('mid point theta')
		h = self.mesh.cell_diameter()
		w = self.test_function('u')
		q = self.test_function('p')
		un = self.solution_function('u')
		pn = self.solution_function('p')
		(u0, p0) = fe.split(self.previous_solution)
		gw = fe.grad(w)
		dw = fe.div(w)

		# Main weak formulation
		Tn = self.stress(un, pn, mu)
		F1 = fe.inner(Tn, gw) + fe.inner(rho * fe.grad(un) * un, w) + q * fe.div(un)
		T0 = self.stress(u0, pn, mu)
		F0 = fe.inner(T0, gw) + fe.inner(rho * fe.grad(u0) * u0, w)
		self.weak_form = fe.inner(rho * (un - u0) / dt, w) + (1 - theta) * F0 + theta * F1
		self.weak_form *= self.dx

		# Determine permeability based on Kozeny-Karman model for packed spheres
		phi = self.porosity
		L_c = self.char_len
		K = L_c ** 2 * phi ** 3 / (150 * (1 - phi) ** 2)
		brinkman_term = 0.5 * mu * I / K * fe.inner(w, un + u0) * self.dx

		self.weak_form += brinkman_term

		# backflow stab
		n = fe.FacetNormal(self.mesh.mesh)
		un_mean = 0.5 * (fe.dot(un, n) - abs(fe.dot(un, n)))
		for id in self.outlet_id_list:
			self.weak_form += (-rho * 0.2 * un_mean * fe.dot(un, w)) * self.ds(id)

		# Add stab if option specify
		if stab:
			self.add_stab()