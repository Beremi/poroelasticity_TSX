def tsx(mesh, days=300, jumps=True, young_e=6e10, poisson_nu=0.2, alpha=0.2, k=6e-19, cpp = 7.712e-12, tau_f=3600.0*12, k1=6e-19, k2=6e-16, k3=6e-13):
    from dolfin import (SubDomain, near, nabla_grad, VectorElement,
                        FiniteElement, MixedElement, FunctionSpace, TrialFunctions,
                        TestFunctions, MeshFunction, Measure, FacetNormal, dx,
                        DirichletBC, inner, div, Constant, Expression, dot,
                        PETScOptions, PETScKrylovSolver, Function, assemble_system,
                        assemble, MPI, UserExpression,
                        norm)
    from mpi4py import MPI as pyMPI
    import numpy as np
    import sys
    from math import sqrt


    com = mpi4py.MPI.COMM_WORLD
    rank = com.rank

    class PermeabilityJumps(UserExpression):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def eval(self, value, x):
            if (x[0] - 50)**2/(1)**2 + (x[1] - 50)**2/(3.5/2 + 0.75/2)**2 <= 1:
                value[0] =  k3
            elif (x[0] - 50)**2/(4.375/2 + 0.75)**2 + (x[1] - 50)**2/(3.5/2 + 0.75)**2 <= 1:
                value[0] = k2
            else:
                value[0] = k1

        def value_shape(self):
            return ()

    def mpi4py_comm(comm):
        """Get mpi4py communicator"""
        try:
            return comm.tompi4py()
        except AttributeError:
            return comm

    def peval(f, x):
        """ Parallel synced eval, taken from
        https://fenicsproject.discourse.group/t/problem-with-evaluation-at-a-point-in-parallel/1188/5
        """
        # hack for function with float as output version in the link has array output
        try:
            yloc = np.array([f(x)])
        except RuntimeError:
            yloc = np.inf * np.array([1])
        comm = mpi4py_comm(f.function_space().mesh().mpi_comm())
        yglob = np.zeros_like(yloc)
        comm.Allreduce(yloc, yglob, op=pyMPI.MIN)
        return yglob

    class Outer(SubDomain):
        def inside(self, x, on_boundary):
            return boundary_outer(x, on_boundary)

    class Inner(SubDomain):
        def inside(self, x, on_boundary):
            return boundary_inner(x, on_boundary)

    def epsilon(u):
        return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

    def boundary_complete(x, on_boundary):
        return on_boundary

    def boundary_outer(x, on_boundary):
        outer = near(x[0], 0) or near(x[0], 100) or \
                near(x[1], 0) or near(x[1], 100)
        return outer and on_boundary

    def boundary_outer_lr(x, on_boundary):
        lr = near(x[0], 0) or near(x[0], 100)
        return lr and on_boundary

    def boundary_outer_bt(x, on_boundary):
        bt = near(x[1], 0) or near(x[1], 100)
        return bt and on_boundary

    def boundary_inner(x, on_boundary):
        outer = near(x[0], 0) or near(x[0], 100) or \
                near(x[1], 0) or near(x[1], 100)
        return on_boundary and not outer

    # physical parameters and timestep, all in SI units
    mu = young_e/(2 * (1 + poisson_nu));
    mu = Constant(mu)
    lmbda = young_e * poisson_nu/((1 + poisson_nu)*(1 - 2*poisson_nu))
    lmbda = Constant(lmbda)
    alpha = Constant(alpha)
    cpp = Constant(cpp)
    tau = Constant(tau_f)

    PC = VectorElement('CG', mesh.ufl_cell(), 2)
    RT = FiniteElement('RT', mesh.ufl_cell(), 2)
    PD = FiniteElement('DG', mesh.ufl_cell(), 1)
    V = MixedElement([PC, RT, PD])
    V = FunctionSpace(mesh, V)
    u, v, p = TrialFunctions(V)
    w, z, q = TestFunctions(V)

    if jumps:
        k = PermeabilityJumps(element=V.sub(2).ufl_element())
    else:
        k = Constant(k)

    boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    outer = Outer()
    inter = Inner()
    outer.mark(boundaries, 1)
    inter.mark(boundaries, 2)
    ds_b = Measure('ds', domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)

    bc_elasticity_lr = DirichletBC(V.sub(0).sub(0), 0, boundary_outer_lr)
    bc_elasticity_bt = DirichletBC(V.sub(0).sub(1), 0, boundary_outer_bt)
    bc = [bc_elasticity_lr, bc_elasticity_bt]

    a_el = 2*mu*inner(epsilon(u), epsilon(w))*dx + lmbda*div(u)*div(w)*dx
    a_cg = a_el + alpha*p*div(w)*dx + \
           tau/k*inner(v, z)*dx + tau*p*div(z)*dx + \
           alpha*div(u)*q*dx + tau*div(v)*q*dx - cpp*p*q*dx

    f = Constant((0, 0))
    g = Constant((0, 0))
    hf = Constant(0)
    p_outer = Constant(3e6)
    p_init = Constant(3e6)
    sigma = Constant(np.array([[-45e6, 0], [0, -11e6]]))

    p_inner = Expression('3e6*max(0.0, 1 - t/(17.0*24.0*3600.0))', t=tau_f, degree=4)  # TODO: lower the degree?
    sigma_factor = Expression('min(1.0, t/(17.0*24.0*3600.0))', t=tau_f, degree=4)

    # scaled rhs, relates to the original by b_v = tau*b_v; b_p = tau*b_p
    rhs = inner(f, w)*dx + tau*inner(g, z)*dx + tau*hf*q*dx
    rhs += -tau*inner(n, z)*p_outer*ds_b(1) - tau*inner(n, z)*p_inner*ds_b(2)
    rhs += cpp*p_init*q*dx
    # initial displacement is zero, nothing is added to rhs
    rhs += -sigma_factor*dot(dot(sigma, n), w)*ds_b(2)


    A, b = assemble_system(a_cg, rhs, bc)

    solver2 = PETScKrylovSolver()
    solver2.set_operators(A, A)
    PETScOptions.set('ksp_type', 'preonly')
    PETScOptions.set('pc_type', 'lu')
    PETScOptions.set('pc_factor_mat_solver_type', 'mumps')
    solver2.set_from_options()

    x2 = Function(V)

    time_steps = []
    iters = []

    POINT_MES = {'hgt15': (50, 50 + 3.5/2 + 1.5),
                 'hgt14': (50, 50 + 3.5/2 + 4.0),
                 'hgt25': (50 + 4.375/2 + 4.0, 50),
                 'hgt24': (50 + 4.375/2 + 1.5, 50)}
    results = {point: [] for point in POINT_MES.keys()}

    # first timestep
    # direct solve
    solver2.solve(x2.vector(), b)
    x_u, x_v, x_p = x2.split()

    time_steps.append(tau_f / (24 * 3600))  # time in days
    for point in POINT_MES.keys():
        results[point].append(-peval(x_p, POINT_MES[point])[0])

    tstep = 2*days  # total numbers of steps
    current_time = tau_f
    for i in range(2, tstep + 1):
        current_time = i * tau_f
        if MPI.comm_world.rank == 0:
            print('Timestep: {}, current time: {} days'.format(i, current_time / (24 * 3600)))
            sys.stdout.flush()

        p_inner.t = current_time
        sigma_factor.t = current_time

        # scaled rhs, relates to the original by b_v = tau*b_v; b_p = tau*b_p
        rhs = inner(f, w)*dx + tau*inner(g, z)*dx + tau*hf*q*dx
        rhs += -sigma_factor*dot(dot(sigma, n), w)*ds_b(2)
        rhs += -tau*inner(n, z)*p_outer*ds_b(1) - tau*inner(n, z)*p_inner*ds_b(2)
        rhs += -cpp*x_p*q*dx + alpha*div(x_u)*q*dx
        b = assemble(rhs)
        for b_condition in bc:
            b_condition.apply(A, b)

        # LU solve
        solver2.solve(x2.vector(), b)
        x_u, x_v, x_p = x2.split()

        time_steps.append(current_time / (24 * 3600))
        for point in POINT_MES.keys():
            results[point].append(-peval(x_p, POINT_MES[point])[0])

    return results


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from dolfin import Mesh, XDMFFile, plot
    import pickle
    import mpi4py

    mesh = Mesh()
    ff = XDMFFile('test_output.xdmf')
    ff.read(mesh)
    print(mesh.num_vertices())

    OUTFILE = 'test_jumps_'


    results = tsx(mesh, days=100, jumps=False)

    com = mpi4py.MPI.COMM_WORLD
    rank = com.rank

    if rank == 0:
        with open(f'{OUTFILE}.pickle', 'wb') as my_file:
            pickle.dump(results, my_file)

