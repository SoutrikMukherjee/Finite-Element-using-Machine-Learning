import matplotlib.pyplot as plt

def plot_displacements(nodes, displacements, title="Nodal Displacements"):
    plt.figure(figsize=(8,4))
    plt.plot(nodes, displacements, 'o-', label='Displacement')
    plt.xlabel("Node")
    plt.ylabel("Displacement (mm)")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

def plot_stress(strains, stresses, title="Stress-Strain Curve"):
    plt.figure(figsize=(6,4))
    plt.plot(strains, stresses, 'r-')
    plt.xlabel("Strain")
    plt.ylabel("Stress")
    plt.title(title)
    plt.grid()
    plt.show()
