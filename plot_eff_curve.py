import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

eff_vals = [
    0.022738, 0.116216, 0.394108, 0.226506, 0.311843, 0.013889, 0.159311,
    0.114528, 0.051804, 0.064209, 0.063315, 0.034005, 0.027382, 0.024314,
    0.022232, 0.023833, 0.017862, 0.013140, 0.018953, 0.007505, 0.015873,
    0.005464, 0.018420, 0.014544, 0.014290, 0.020109, 0.023948, 0.012350,
    0.013875, 0.022289, 0.003490, 0.006704, 0.021550, 0.001182, 0.006345,
    0.004682, 0.314293, 0.111995, 0.052394, 0.012394, 0.044847, 1.318629,
    0.017949, 0.016586, 0.020631, 0.017345, 0.032121, 0.023111, 0.016470,
    0.021402, 0.022381, 0.011812, 0.005423, 0.034761, 0.019877, 0.010479,
]

# theta après ajout du point n (= modèle utilisé pour calculer EFF(n+1))
# extrait de output_0306_0741.txt — lignes LOO= 1 à 56
theta_fc = [
    0.44257466, 2.36039325, 9.2103542,  3.60979683, 8.10024531,
    1.06458581, 2.47149113, 3.71847186, 3.64381212, 2.41476392,
    2.41833035, 0.73958397, 0.71924181, 0.71499297, 0.72155916,
    0.73683264, 0.7502881,  0.68166919, 0.70412542, 0.72201059,
    0.74707378, 0.75954968, 0.76838902, 0.47016801, 0.47994752,
    0.80273083, 0.49043851, 1.21542266, 0.46414182, 1.1755693,
    1.14425564, 1.15190345, 1.04951423, 1.0611126,  0.53923892,
    0.52886749, 61.71705569,62.33279773,63.19987632,62.30371817,
    61.89973458,38.82129335,36.69255822,34.29842,   1.09859804,
    1.10484466, 1.10581628, 1.11158777, 1.08720969, 58.36018465,
    57.22555352,58.34802231,50.88050508,62.46215722,62.55491284,
    58.2945871,
]

theta_fy = [
    0.84613221, 3.16324024, 0.6579585,  7.0978205,  0.64527869,
    0.52445027, 1.16943017, 2.59035696, 2.5956186,  1.21240335,
    1.211678,   2.62895929, 2.62293047, 2.53803522, 2.53268675,
    2.53951769, 2.51470499, 2.57841315, 2.67050935, 2.74326842,
    2.84799171, 2.86626354, 3.02383371, 3.89830738, 3.91684452,
    3.02829512, 3.9571559,  1.48403356, 2.91076062, 1.60436688,
    1.62981386, 1.45455962, 1.45155465, 1.45556017, 1.58037655,
    1.66958807, 99.98025609,99.98428054,99.99020894,99.93122735,
    99.99957731,59.26636205,55.0347269, 52.52919843, 1.77868691,
    1.77906638, 1.77701454, 1.78626388, 1.77134774, 82.79711929,
    81.62297717,82.6632101, 72.7987043, 79.41039403,79.850051,
    76.71946438,
]

n = list(range(1, len(eff_vals) + 1))
tol = 0.001

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# --- Axe linéaire EFF ---
ax = axes[0]
ax.plot(n, eff_vals, 'o-', color='steelblue', markersize=4, lw=1.2)
ax.axhline(tol, color='red', lw=1.5, ls='--', label=f'tol = {tol}')
ax.set_xlabel('Point EFF ajouté (n)')
ax.set_ylabel('EFF')
ax.set_title('EFF par itération — run 0306_0741 (sans protections kernel)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(1, len(n))

# --- Axe log EFF ---
ax2 = axes[1]
ax2.semilogy(n, eff_vals, 'o-', color='darkorange', markersize=4, lw=1.2)
ax2.axhline(tol, color='red', lw=1.5, ls='--', label=f'tol = {tol}')
ax2.set_xlabel('Point EFF ajouté (n)')
ax2.set_ylabel('EFF (log)')
ax2.set_title('EFF par itération — échelle log')
ax2.legend()
ax2.grid(True, which='both', alpha=0.3)
ax2.set_xlim(1, len(n))

# --- Theta (log) ---
ax3 = axes[2]
ax3.semilogy(n, theta_fc, 'o-', color='seagreen',  markersize=4, lw=1.2, label=r'$\theta_{fc}$')
ax3.semilogy(n, theta_fy, 's-', color='mediumpurple', markersize=4, lw=1.2, label=r'$\theta_{fy}$')
ax3.set_xlabel('Point EFF ajouté (n)')
ax3.set_ylabel('θ (log)')
ax3.set_title('Hyperparamètre θ GEPCK après ajout du point n')
ax3.legend()
ax3.grid(True, which='both', alpha=0.3)
ax3.set_xlim(1, len(n))

plt.tight_layout()
out = r"C:\_workingDir\_SF\test flexion\output\plot_eff_curve_0306_0741.png"
plt.savefig(out, dpi=130)
print(f"Sauvé : {out}")
