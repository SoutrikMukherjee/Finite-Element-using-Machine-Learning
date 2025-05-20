import numpy as np

def element_stiffness(E, A, L):
    """Returns the local stiffness matrix for a 1D bar element."""
    k = (E * A) / L
    return k * np.array([[1, -1], [-1, 1]])

def assemble_global_stiffness(num_elements, E, A, L):
    """Assembles global stiffness matrix for a 1D rod with fixed nodes."""
    size = num_elements + 1
    K = np.zeros((size, size))
    for i in range(num_elements):
        k = element_stiffness(E, A, L)
        K[i:i+2, i:i+2] += k
    return K

def apply_boundary_conditions(K, F, fixed_nodes):
    """Apply boundary conditions to stiffness matrix and force vector."""
    K_mod = K.copy()
    F_mod = F.copy()
    for node in fixed_nodes:
        K_mod[node, :] = 0
        K_mod[:, node] = 0
        K_mod[node, node] = 1
        F_mod[node] = 0
    return K_mod, F_mod

def solve_displacements(K, F):
    """Solve for nodal displacements."""
    return np.linalg.solve(K, F)
