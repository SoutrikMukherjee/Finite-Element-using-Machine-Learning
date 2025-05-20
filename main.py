# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from fe_utils import assemble_global_stiffness, apply_boundary_conditions, solve_displacements
from visualization import plot_displacements, plot_stress
from ml_failure import generate_synthetic_data, train_failure_predictor, predict_failure

# 1. FEA SETUP (1D rod, just for demo)
num_elements = 5
E = 200e3      # Young's modulus (MPa)
A = 10.0       # Area (mm^2)
L = 100.0      # Length of each element (mm)
nodes = np.linspace(0, L*num_elements, num_elements+1)
F = np.zeros(num_elements+1)
F[-1] = 1000   # Point load at last node

# 2. FEA CALC
K = assemble_global_stiffness(num_elements, E, A, L)
K_bc, F_bc = apply_boundary_conditions(K, F, fixed_nodes=[0])
displacements = solve_displacements(K_bc, F_bc)
print("Displacements (mm):", displacements)
plot_displacements(nodes, displacements)

# 3. Generate stress/strain data for plotting
strain = displacements[1:] / L
stress = E * strain
plot_stress(strain, stress)

# 4. ML: Failure Prediction
X, y = generate_synthetic_data()
clf = train_failure_predictor(X, y)
# Predict on new data (e.g., measured stress/strain)
test_points = np.column_stack([strain, stress, np.zeros_like(strain)])
failure_preds = predict_failure(clf, test_points)
print("Failure Prediction for Each Element:", failure_preds)

# 5. Data Science: Pandas Analysis
df = pd.DataFrame({'Node': range(1, len(strain)+1), 'Strain': strain, 'Stress': stress, 'Failure': failure_preds})
print("\nSummary Table:\n", df)

# Detect anomalies: flag any element with predicted failure or high strain
df['Anomaly'] = (df['Failure'] == 1) | (df['Strain'] > 0.008)
print("\nAnomalous Elements:\n", df[df['Anomaly']])
