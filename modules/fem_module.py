import fenics as fe
import numpy as np
import mshr as mesher
import sympy 
import os
import re
from scipy.optimize import least_squares as lsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FEMSimulation():

#==================================================================
    def __init__(self, storage_dir = None):

        self.current_dir  = os.getcwd()

        self.storage = os.path.join(self.current_dir, 'storage')

        self.fast_run  = None
        self.mode      = 'test'
        self.dimension = 2

        self.initial_time = 0
        self.final_time   = None
        self.current_time = None

        self.diffusion_coefficient = 1.0
        self.lam = 1
        self.alpha = 1
        self.beta = 1

        self.dt           = 0.05

        self.cell_exact_str    = None
        self.cell_rhs_fun_str  = None
        self.oxygen_exact_str    = None
        self.oxygen_rhs_fun_str  = None

        self.cell_boundary_fun = None
        self.cell_boundary_conditions = None

        self.oxygen_boundary_fun = None
        self.oxygen_boundary_conditions = None

        self.mesh         = None

        self.cell         = None
        self.oxygen       = None
        self.cell_n       = None

        self.cell_rhs_fun   = None
        self.oxygen_rhs_fun = None

        self.cell_ic_fun  = None
        self.oxygen_ic_fun= None

        self.cell_bilinear_form  = None
        self.oxygen_bilinear_form= None

        self.cell_rhs     = None
        self.oxygen_rhs   = None

        self.function_space = None

        self.oxygen_vtkfile = None
        self.cell_vtkfile   = None

#==================================================================

#==================================================================
    def set_parameters(self):
        pass

#==================================================================
    def create_initial_condition_function(self):

        if self.mode != 'test':
            pass

            self.cell_ic_fun =\
                    fe.Expression(self.cell_exact_str, degree=2,\
                    alpha = self.alpha, beta = self.beta, t = 0)

            self.oxygen_ic_fun =\
                    fe.Expression(self.oxygen_exact_str, degree=2,\
                    alpha = self.alpha, beta = self.beta, t = 0)

#==================================================================
    def create_exact_solution_and_rhs_fun_strings(self):

        if self.mode != 'test':
            return

        print('Creating exact solution and rhs strings')

        x,y,a,b,l,t = sympy.symbols('x[0], x[1], alpha, beta, lam, t')

        if self.dimension == 2: 
            cell_exact = l*x*y*t 
            oxygen_exact = 1 + x**2 + a * y**2 + b * t

        if self.dimension == 1: 
            cell_exact = l*x*t 
            oxygen_exact = 1 + a * x**2 + b * t

        '''
        cell function
        '''
        cell_t = cell_exact.diff(t)
        rhs_fun_cell = cell_t - oxygen_exact * cell_exact * (1 - cell_exact)

        if self.dimension == 1: 
            '''
            oxygen function
            '''
            diffusion_term = oxygen_exact.diff(x,2)

        if self.dimension == 2: 
            '''
            oxygen function
            '''
            grad_oxygen = sympy.Matrix([oxygen_exact]).jacobian([x,y]).T
            diffusion_term = grad_oxygen.jacobian([x,y]).trace()

        rhs_fun_oxygen = diffusion_term - oxygen_exact * cell_exact

        self.cell_exact_str   = sympy.printing.ccode(cell_exact)
        self.cell_rhs_fun_str = sympy.printing.ccode(rhs_fun_cell)

        self.oxygen_exact_str   = sympy.printing.ccode(oxygen_exact)
        self.oxygen_rhs_fun_str = sympy.printing.ccode(rhs_fun_oxygen)

#==================================================================
    def create_rhs_fun(self):

        if self.mode == 'test': 

            print('Creating rhs function')
            self.cell_rhs_fun =\
                    fe.Expression(self.cell_rhs_fun_str, degree=2,\
                    alpha = self.alpha,\
                    beta  = self.beta,\
                    lam   = self.lam,\
                    t     = 0)

            self.oxygen_rhs_fun =\
                    fe.Expression(self.oxygen_rhs_fun_str, degree=2,\
                    alpha = self.alpha,\
                    beta  = self.beta,\
                    lam   = self.lam,\
                    t     = 0)
        else:
            '''
            Zero RHS for the experimental case
            '''
            self.cell_rhs_fun   = fe.Constant(0)
            self.oxygen_rhs_fun = fe.Constant(0)

#==================================================================
    def create_boundary_conditions(self):

        if self.mode == 'test':

            print('Creating boundary function')
            self.cell_boundary_fun =\
                    fe.Expression(self.cell_exact_str, degree=2,\
                    alpha = self.alpha, beta = self.beta, t = 0)

            self.oxygen_boundary_fun =\
                    fe.Expression(self.oxygen_exact_str, degree=2,\
                    alpha = self.alpha, beta = self.beta, t = 0)

        else:
            '''
            Homogeneous Neumann conditions
            Cells do not escape.
            '''
            #self.cell_boundary_fun = fe.Constant(0)

            self.oxygen_boundary_fun = fe.Constant(1)


        def is_on_the_boundary(x, on_boundary):
            return on_boundary

        self.oxygen_boundary_conditions =\
                fe.DirichletBC(\
                self.function_space,\
                self.oxygen_boundary_fun,\
                is_on_the_boundary)

#==================================================================
    def create_simple_mesh(self):

        #domain      = mesher.Circle(fe.Point(0,0), 1)
        #mesh        = mesher.generate_mesh(domain, 64)
        print('Creating simple mesh')
        nx = ny = 8

        if self.dimension == 1: 
            self.mesh = fe.UnitIntervalMesh(nx)
            #self.mesh = fe.IntervalMesh(nx,-4, 4)


        if self.dimension == 2: 
            self.mesh = fe.UnitSquareMesh(nx, ny)

        '''
        finite_element         = self.function_space.element()
        map_cell_index_to_dofs = self.function_space.dofmap()
        for cell in fe.cells(self.mesh):
            print(map_cell_index_to_dofs.cell_dofs(cell.index()))
            print(finite_element.tabulate_dof_coordinates(cell))
            print('------------')
            break
        '''

#==================================================================
    def create_mesh(self):

        if self.mode == 'test':
            self.create_simple_mesh()
        else:
            self.create_simple_mesh()

#==================================================================
    def set_function_spaces(self):

        self.function_space = fe.FunctionSpace(self.mesh, 'P', 1)

#==================================================================
    def compute_error(self):

        if self.mode != 'test':
            return

        error_L2 = fe.errornorm(self.boundary_fun, self.cell_n, 'L2')
        error_LI = np.abs(\
                fe.interpolate(\
                self.boundary_fun,self.function_space).vector().get_local() -\
                self.cell_n.vector().get_local()\
                ).max()

        print('L2 error at t = {:.3f}: {:.2e}'.format(\
                self.current_time, error_L2))

        print('LI error at t = {:.3f}: {:.2e}'.format(\
                self.current_time, error_LI))

        self.error_list.append(error_L2) 


#==================================================================
    def set_initial_conditions(self):

        self.current_time = self.initial_time

        #Initial condition
        #self.cell_n = fe.project(self.boundary_fun, self.function_space)

        if self.mode == 'test':

            print('Setting initial conditions')

            self.cell_boundary_fun.t = self.current_time
            self.cell_n =\
                    fe.interpolate(self.cell_boundary_fun,\
                    self.function_space)

            self.oxygen_boundary_fun.t = self.current_time
            self.oxygen_n =\
                    fe.interpolate(self.oxygen_boundary_fun,\
                    self.function_space)

        else:

            self.cell_n   =\
                    fe.project(self.cell_ic_fun, self.function_space)

            self.oxygen_n =\
                    fe.project(self.oxygen_ic_fun, self.function_space)

        self.cell   = fe.Function(self.function_space)
        self.oxygen = fe.Function(self.function_space)

        self.compute_error()
        self.save_snapshot()

#==================================================================
    def create_bilinear_forms_and_rhs(self):

        self.create_oxygen_bilinear_form()
        self.create_cell_bilinear_form()



#==================================================================
    def create_cell_bilinear_form(self):


        # Define variational problem
        u = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)

        self.bilinear_form_u = (1 + self.dt * 0) * (u * v * fe.dx)
        self.rhs_u = ( self.cell_n +\
                self.dt * (self.c * self.cell_n * (1 - self.cell_n) +\
                self.rhs_fun_u) ) * v * fe.dx



#==================================================================
    def create_oxygen_bilinear_form(self):


        # Define variational problem
        u = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)

        self.bilinear_form_oxygen = self.diffusion_coefficient *\
                fe.dot(fe.grad(u), fe.grad(v)) * fe.dx

        self.rhs_oxygen = (self.cell_n * self.oxygen_n + self.rhs_fun_oxygen) * v * fe.dx


#==================================================================
    def solve_problem(self):

        fe.solve(self.bilinear_form_u == self.rhs_u,\
                self.cell, self.boundary_conditions_u)

        fe.solve(self.bilinear_form_oxygen == self.rhs_oxygen,\
                self.oxygen, self.boundary_conditions_oxygen)

#==================================================================
    def set_data_dirs(self):

        txt = 'oxygen.pvd'
        fname = os.path.join(self.fem_storage_dir, txt)
        self.oxygen_vtkfile = fe.File(fname)

        txt = 'cell.pvd'
        fname = os.path.join(self.fem_storage_dir, txt)
        self.cell_vtkfile = fe.File(fname)

#==================================================================
    def save_snapshot(self):

        if self.fast_run:
            return

        self.oxygen_vtkfile << (self.oxygen, self.current_time)


#==================================================================
    def run(self):

        self.create_exact_solution_and_rhs_fun_strings()
        self.create_initial_condition_function()
        self.create_rhs_fun()
        self.create_mesh()
        self.set_function_spaces()
        self.create_boundary_conditions()
        self.set_initial_conditions()
        self.create_bilinear_forms_and_rhs()
        print('Alles gut')
        exit()

        while self.current_time < self.final_time: 
            
            self.current_time += self.dt
            print('t = {:0.2f}'.format(self.current_time))
            self.boundary_fun.t = self.current_time
            self.rhs_fun.t      = self.current_time
            self.solve_problem()
            self.cell_n.assign(self.cell)
            self.compute_error()
            self.save_snapshot()

            print('------------------------')
            
        
        print('Alles ist gut')
        



