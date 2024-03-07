import ufl
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc  # needed to carry out the init :(
import dolfinx as dfx
from dolfinx.fem import Constant
from ufl import grad, inner, div, dx
import matplotlib.pyplot as plt


SEC_IN_DAY = 24 * 60 * 60

# Tunnel ellipse, part of the input, efficiently mesh information
TUNNEL_X_HALF_AXIS = 4.375 / 2
TUNNEL_Y_HALF_AXIS = 3.5 / 2

# Parameters of the inner ellipses, guesstimated to match Standa's drawing, see picture in NLAA paper
INNER_X_HALF_AXIS = 1
INNER_Y_HALF_AXIS = 3.5 / 2 + 0.75 / 2

OUTER_X_HALF_AXIS = 4.375 / 2 + 0.75
OUTER_Y_HALF_AXIS = 3.5 / 2 + 0.75

# just for output, to be deleted in final version
XDMF_OUT = False
PICTURE_DATA = True


def epsilon(u):
    return ufl.sym(ufl.nabla_grad(u))


def boundary_outer(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], -50), np.isclose(x[0], 50)),
                         np.logical_or(np.isclose(x[1], -50), np.isclose(x[1], 50)))


def boundary_outer_lr(x):
    return np.logical_or(np.isclose(x[0], -50), np.isclose(x[0], 50))


def boundary_outer_bt(x):
    return np.logical_or(np.isclose(x[1], -50), np.isclose(x[1], 50))


def boundary_inner(x):
    return np.isclose(x[0]**2 / TUNNEL_X_HALF_AXIS**2 + x[1]**2 / TUNNEL_Y_HALF_AXIS**2, 1.0)


# "Marker" true/false function for the subdomains
def innermost_ellipse(x):
    return x[0]**2 / INNER_X_HALF_AXIS**2 + x[1]**2 / INNER_Y_HALF_AXIS**2 <= 1


def outer_ellipse(x):
    return np.logical_and(np.logical_not(innermost_ellipse(x)),
                          (x[0]**2 / OUTER_X_HALF_AXIS**2 + x[1]**2 / OUTER_Y_HALF_AXIS**2 <= 1))


def undisturbed_massif(x):
    return np.logical_and(np.logical_not(innermost_ellipse(x)), np.logical_not(outer_ellipse(x)))


def produce_midpoint_cell_numbers(mesh):
    # https://fenicsproject.discourse.group/t/application-of-point-forces-mapping-vertex-indices-to-corresponding-dofs/9646/2
    # TODO: osklivy hack, nutno se zeptat na foru nebo jinde, jak to resit dobre
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local +\
        mesh.topology.index_map(mesh.topology.dim).num_ghosts
    midpoints = dfx.mesh.compute_midpoints(mesh, 2, list(range(num_cells)))
    # TODO: check if this really works, cells numbering might be ordered differently
    inner, inner2, outer, massif = [], [], [], []
    for i, (x, y, _) in enumerate(midpoints):
        if x**2 / INNER_X_HALF_AXIS**2 + y**2 / INNER_Y_HALF_AXIS**2 <= 1:
            inner.append(i)
        elif x**2 / INNER_Y_HALF_AXIS**2 + y**2 / INNER_X_HALF_AXIS**2 <= 1:
            inner2.append(i)
        elif x**2 / OUTER_X_HALF_AXIS**2 + y**2 / OUTER_Y_HALF_AXIS**2 <= 1:
            outer.append(i)
        else:
            massif.append(i)
    return inner, inner2, outer, massif


def tsx(mesh, lmbda, mu, alpha, cpp, cpp_f_inner, cpp_f_inner2, cpp_f_outer, k_massif,
        k_inner1_ellipse, k_inner2_ellipse, k_outer_ellipse, tau, t_steps_num,
        sigma_xx=-45e6, sigma_yy=-11e6):
    lmbda = Constant(mesh, lmbda)
    mu = Constant(mesh, mu)
    alpha = Constant(mesh, alpha)
    cpp_f = cpp
    # cpp = Constant(mesh, cpp)
    tau_f = tau  # useful to have around as float
    tau = Constant(mesh, tau)  # useful for matrix assembly

    pressure_init = 2697750.0  # water pressure in the massive, initial and outer condition for pressure

    # Spaces and functions
    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V_element = ufl.MixedElement(P2, P1)
    V = dfx.fem.FunctionSpace(mesh, V_element)
    u, p = ufl.TrialFunctions(V)
    w, q = ufl.TestFunctions(V)
    x_h = dfx.fem.Function(V)
    DG = dfx.fem.FunctionSpace(mesh, ufl.FiniteElement('DG', mesh.ufl_cell(), 0))  # TODO: use better function space?

    k_func = dfx.fem.Function(DG)
    cpp_func = dfx.fem.Function(DG)
    in1, in1b, in2, massif = produce_midpoint_cell_numbers(mesh)
    k_func.x.array[in1] = k_inner1_ellipse
    k_func.x.array[in1b] = k_inner2_ellipse
    k_func.x.array[in2] = k_outer_ellipse
    k_func.x.array[massif] = k_massif
    cpp_func.x.array[in1] = cpp_f_inner
    cpp_func.x.array[in1b] = cpp_f_inner2
    cpp_func.x.array[in2] = cpp_f_outer
    cpp_func.x.array[massif] = cpp_f

    # subdomains implemented as a modification of a measure, can be precomputed and used as input
    # https://fenicsproject.discourse.group/t/define-bilinear-form-with-subdomains/11943
    domain_functions = {
        1: innermost_ellipse,
        2: outer_ellipse,
        3: undisturbed_massif
    }
    domain_indices, domain_markers = [], []
    for tag in domain_functions:
        entities = dfx.mesh.locate_entities(mesh, 2, domain_functions[tag])
        domain_indices.append(entities)
        domain_markers.append(np.full_like(entities, tag))

    domain_indices = np.hstack(domain_indices).astype(np.int32)
    domain_markers = np.hstack(domain_markers).astype(np.int32)
    sorted_domain = np.argsort(domain_indices)
    subdomain_tag = dfx.mesh.meshtags(mesh, 2, domain_indices[sorted_domain], domain_markers[sorted_domain])
    # dx = ufl.Measure('dx', domain=mesh, subdomain_data=subdomain_tag)

    # infrastructure for solution evaluation, can be precomputed and used as input
    bb_tree = dfx.geometry.bb_tree(mesh, mesh.topology.dim)
    evaluation_points = np.zeros((3, 4))  # must be 3D, for some reason
    evaluation_points[0, :] = [0, 0, 0 + 4.375 / 2 + 4.0, 0 + 4.375 / 2 + 1.5]
    evaluation_points[1, :] = [0 + 3.5 / 2 + 1.5, 0 + 3.5 / 2 + 4.0, 0, 0]
    cells_t = []
    points_on_proc_t = []
    cell_candidates_t = dfx.geometry.compute_collisions_points(bb_tree, evaluation_points.T)
    colliding_cells_t = dfx.geometry.compute_colliding_cells(mesh, cell_candidates_t, evaluation_points.T)
    for i, point in enumerate(evaluation_points.T):
        if len(colliding_cells_t.links(i)) > 0:
            points_on_proc_t.append(point)
            cells_t.append(colliding_cells_t.links(i)[0])

    ready_eval_points = np.array(points_on_proc_t, dtype=np.float64)

    # bc -> zero normal displacements, 3e6 outer pressure and 3e6 to zero pressure in tunnel
    # all of this can be also precomputed and used as input
    # Dirichlet
    # TODO: use dict and cycle
    lr_edges = dfx.mesh.locate_entities_boundary(mesh, 1, boundary_outer_lr)
    dofs_displacement_lr = dfx.fem.locate_dofs_topological(V.sub(0).sub(0), 1, lr_edges)
    bc_elastic_lr = dfx.fem.dirichletbc(Constant(mesh, 0.0), dofs_displacement_lr, V.sub(0).sub(0))

    bt_edges = dfx.mesh.locate_entities_boundary(mesh, 1, boundary_outer_bt)
    dofs_displacement_bt = dfx.fem.locate_dofs_topological(V.sub(0).sub(1), 1, bt_edges)
    bc_elastic_bt = dfx.fem.dirichletbc(Constant(mesh, 0.0), dofs_displacement_bt, V.sub(0).sub(1))

    outer_edges = dfx.mesh.locate_entities_boundary(mesh, 1, boundary_outer)
    dofs_pressure_outer = dfx.fem.locate_dofs_topological(V.sub(1), 1, outer_edges)
    bc_pressure_outer = dfx.fem.dirichletbc(Constant(mesh, pressure_init), dofs_pressure_outer, V.sub(1))

    pbc_expression = Constant(mesh, pressure_init * max(0.0, 1 - tau_f / (17 * SEC_IN_DAY)))
    inner_edges = dfx.mesh.locate_entities_boundary(mesh, 1, boundary_inner)
    dofs_pressure_inner = dfx.fem.locate_dofs_topological(V.sub(1), 1, inner_edges)
    bc_pressure_inner = dfx.fem.dirichletbc(pbc_expression, dofs_pressure_inner, V.sub(1))

    bcs = [bc_elastic_lr, bc_elastic_bt, bc_pressure_inner, bc_pressure_outer]

    # Neumann - not needed at all, potentionally usefull for 3 field formulation, can be deleted for production
    facets = dfx.mesh.locate_entities_boundary(mesh, 1, boundary_inner)
    facet_marker = np.full_like(facets, 1)
    facet_tag = dfx.mesh.meshtags(mesh, 1, facets, facet_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
    n = ufl.FacetNormal(mesh)

    # form construction
    sigma_init = Constant(mesh, np.array([[sigma_xx, 0], [0, sigma_yy]]))
    sigma_expression = Constant(mesh, min(1.0, tau_f / (17 * SEC_IN_DAY)))

    # TODO: BEWARE CPP
    cpp = cpp_func

    ff_term = cpp / tau * p * q * dx  # flux-flux term
    ff_term += k_func * inner(grad(p), grad(q)) * dx
    # for tag in domain_values:
    #     ff_term += domain_values[tag]*inner(grad(p), grad(q))*dx(tag)

    a = dfx.fem.form(2 * mu * inner(epsilon(u), epsilon(w)) * dx + lmbda * div(u) * div(w) * dx - alpha * p * div(w) * dx +
                     alpha / tau * q * div(u) * dx + ff_term)

    # rhs construction
    f = Constant(mesh, (0.0, 0.0))  # elastic volume force
    g = Constant(mesh, 0.0)  # pressure volume force

    L = dfx.fem.form(inner(f, w) * dx + g * q * dx +
                     cpp / tau * pressure_init * q * dx -
                     sigma_expression * inner(sigma_init, epsilon(w)) * dx)

    # assembly and set bcs
    A = dfx.fem.petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()
    b = dfx.fem.petsc.assemble_vector(L)
    dfx.fem.petsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dfx.fem.set_bc(b, bcs)

    # PETSc4py section with solver setup
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A, A)
    solver.setType('preonly')
    solver.getPC().setType('lu')
    opts = PETSc.Options()
    opts['pc_factor_mat_solver_type'] = 'mumps'
    solver.setFromOptions()

    # solve for first timestep
    solver.solve(b, x_h.vector)
    x_h.x.scatter_forward()
    u_h, p_h = x_h.split()

    if PICTURE_DATA:
        pressure_values = []
        pressure_values.append(p_h.eval(ready_eval_points, cells_t))

    if XDMF_OUT:
        pressure_file = dfx.io.XDMFFile(mesh.comm, 'results_test.xdmf', 'w')
        pressure_file.write_mesh(mesh)
        pressure_file.write_function(k_func, tau_f)
        # pressure_file.write_meshtags(subdomain_tag, mesh.geometry)
        # pressure_file.write_function(p_h, tau_f/SEC_IN_DAY)

    # Time loop
    # TODO: Tie all of the code into the big time loop
    current_time = tau_f
    for _ in range(2, t_steps_num):
        current_time += tau_f
        sigma_expression.value = min(1.0, current_time / (17 * SEC_IN_DAY))
        pbc_expression.value = pressure_init * max(0.0, 1 - current_time / (17 * SEC_IN_DAY))
        L = dfx.fem.form(inner(f, w) * dx + g * q * dx +
                         alpha / tau * div(u_h) * q * dx + cpp / tau * p_h * q * dx -
                         sigma_expression * inner(sigma_init, epsilon(w)) * dx)

        b = dfx.fem.petsc.assemble_vector(L)
        dfx.fem.petsc.apply_lifting(b, [a], [bcs])  # ???
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        dfx.fem.set_bc(b, bcs)
        solver.solve(b, x_h.vector)
        x_h.x.scatter_forward()
        u_h, p_h = x_h.split()

        if XDMF_OUT:
            # pressure_file.write_function(p_h, current_time/SEC_IN_DAY)
            pass
        if PICTURE_DATA:
            pressure_values.append(p_h.eval(ready_eval_points, cells_t))
    if XDMF_OUT:
        pressure_file.close()

    return pressure_values


class WrapperTOM:
    def __init__(self):
        self.young_e = 6e10
        self.poisson_nu = 0.2
        self.mu = self.young_e / (2 * (1 + self.poisson_nu))
        self.lmbda = self.young_e * self.poisson_nu / ((1 + self.poisson_nu) * (1 - 2 * self.poisson_nu))
        self.alpha = 0.2
        self.cpp = 7.712e-12
        self.timestep = float(SEC_IN_DAY)

        meshfile_name = 'tsx_inner_layer_shifted.xdmf'
        # meshfile_name = 'tsx_nlaa_mesh_shifted.xdmf'
        with dfx.io.XDMFFile(MPI.COMM_WORLD, meshfile_name, 'r') as mesh_file:
            self.mesh = mesh_file.read_mesh(name='Grid')

        # data = tsx(mesh, lmbda, mu, alpha, cpp, 6e-19, 6e-19, 6e-19, timestep,
        #             t_steps_num=800)
        # data_fp = np.zeros((4, len(data)))
        # for i, item in enumerate(data):
        #     data_fp[:, i] = [value[0] for value in data[i]]

    def run(self, params):
        young_e = np.exp(params[0])
        cpp1 = np.exp(params[1])
        cpp2 = np.exp(params[2])
        cpp3 = np.exp(params[3])
        cpp4 = np.exp(params[4])
        kf1 = np.exp(params[5])
        kf2 = np.exp(params[6])
        kf3 = np.exp(params[7])
        kf4 = np.exp(params[8])
        sigma_xx = np.exp(params[9])
        sigma_yy = np.exp(params[10])

        poisson_nu = 0.2
        mu = young_e / (2 * (1 + poisson_nu))
        lmbda = young_e * poisson_nu / ((1 + poisson_nu) * (1 - 2 * poisson_nu))

        data2 = tsx(self.mesh, lmbda, mu, self.alpha, cpp1, cpp2, cpp3, cpp4, kf1, kf2, kf3, kf4, self.timestep,
                    t_steps_num=365, sigma_xx=-sigma_xx, sigma_yy=-sigma_yy)
        data2_fp = np.zeros((4, len(data2)))
        for i, item in enumerate(data2):
            data2_fp[:, i] = [value[0] for value in data2[i]]

        return data2_fp
