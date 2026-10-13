import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson

# ======================== Plate Properties ====================== #

a = 0.508
E = 70e9
nu = 0.33
h = 0.00127
rho = 2700

Dp = (E * h**3) / (12 * (1 - nu**2))
Nx0 = 0.0
n_modes = 6
mode_numbers = np.arange(1, n_modes + 1)
N = 101
x = np.linspace(0.0, a, N)


# ========================== Flow Properties =========================== #
M_inf = 2
rho_inf = 1.225
gamma_air = 1.4
R_air = 287.05
T_inf = 288.15

a_inf = np.sqrt(gamma_air * R_air * T_inf)
U_inf = M_inf * a_inf

dyn_P_inf = 0.5 * rho_inf * U_inf**2
cavity_pressure_delta = 0.0


# ==================== Simply Supported Shape Functions ===================== #

phi = np.sin(np.outer(mode_numbers * np.pi / a, x))
dphi_dx = (mode_numbers[:, None] * np.pi / a) * np.cos(np.outer(mode_numbers * np.pi / a, x))


def x_deriv(q):
    return q @ dphi_dx


def t_deriv(dq_dt):
    return dq_dt @ phi


def Mass_Vector():
    return 0.5 * rho * h * a * np.ones(n_modes)


def Stiffness_Vector(Nx):
    k = Dp * ((mode_numbers * np.pi / a) ** 4) * (a / 2.0)
    k -= Nx * ((mode_numbers * np.pi / a) ** 2) * (a / 2.0)
    return k


def Aero_Modal_Matrices(M, dyn_p_test):
    beta = (M**2 - 2.0) / (M**2 - 1.0)
    const = -2.0 * dyn_p_test / np.sqrt(M**2 - 1.0)

    aero_stiffness = np.zeros((n_modes, n_modes), dtype=float)
    aero_damping = np.zeros((n_modes, n_modes), dtype=float)

    for i in range(n_modes):
        for j in range(n_modes):
            aero_stiffness[i, j] = const * simpson(phi[i, :] * dphi_dx[j, :], x=x)
            aero_damping[i, j] = const * (beta / U_inf) * simpson(phi[i, :] * phi[j, :], x=x)

    return aero_stiffness, aero_damping


def Aero_Modal_Force(q, dq_dt):
    const = -2.0 * dyn_P_inf / np.sqrt(M_inf**2 - 1.0)
    beta = (M_inf**2 - 2.0) / (M_inf**2 - 1.0)

    dw_dx = x_deriv(q)
    dw_dt = t_deriv(dq_dt)
    del_p = const * (dw_dx + (beta / U_inf) * dw_dt) + cavity_pressure_delta

    return np.array([simpson(del_p * phi[s, :], x=x) for s in range(n_modes)])


def flutter_growth_rate(M, dyn_P_test):
    Ms_local = Mass_Vector()
    Ks_local = Stiffness_Vector(Nx0)
    M_diag = np.diag(Ms_local)
    K_diag = np.diag(Ks_local)
    A_aero, C_aero = Aero_Modal_Matrices(M, dyn_P_test)

    M_inv = np.linalg.inv(M_diag)
    state_matrix = np.block([
        [np.zeros((n_modes, n_modes)), np.eye(n_modes)],
        [M_inv @ (A_aero - K_diag), M_inv @ C_aero],
    ])

    eigvals = np.linalg.eigvals(state_matrix)
    return np.max(np.real(eigvals))


def flutter_sweep_vs_dynamic_pressure(M, dyn_P_values=None, dyn_P_min=1.0e4, dyn_P_max=5.0e4, n_points=400):
    dyn_P_values = np.linspace(dyn_P_min, dyn_P_max, n_points)

    growth = np.array([flutter_growth_rate(M, dyn_P) for dyn_P in dyn_P_values])

    q_dyn_flutter = None
    for i in range(len(dyn_P_values) - 1):
        if growth[i] <= 0.0 and growth[i + 1] > 0.0:
            g0 = growth[i]
            g1 = growth[i + 1]
            q_dyn_flutter = np.interp(0.0, [g0, g1], [dyn_P_values[i], dyn_P_values[i + 1]])
            break

    return dyn_P_values, growth, q_dyn_flutter

 
Ms = Mass_Vector()
Ks = Stiffness_Vector(Nx0)


def state_space(t, y):
    dqn_dt = y[0:n_modes]
    qn = y[n_modes:]

    dw_dx = x_deriv(qn)
    Nx_nonlinear = E * h / (2.0 * a) * simpson(dw_dx**2, x=x)
    K_nonlinear = Nx_nonlinear * ((mode_numbers * np.pi / a) ** 2) * (a / 2.0)

    F = Aero_Modal_Force(qn, dqn_dt)
    ddqn_ddt = (F - (Ks * qn + K_nonlinear * qn)) / Ms

    return np.concatenate((ddqn_ddt, dqn_dt))


def solve_system(dq0_input, q0_input):
    y0 = np.concatenate((dq0_input, q0_input))
    t0 = 0.0
    tf = 1.0
    t_series = np.linspace(t0, tf, 5000)
    return solve_ivp(state_space, (t0, tf), y0, method='LSODA', t_eval=t_series, rtol=1e-8, atol=1e-10)


q0 = np.zeros(n_modes)
dq0 = np.zeros(n_modes)
q0[0] = h * 1.0e0                               # Function of plate thickness

solution = solve_system(dq0, q0)
qn = solution.y[n_modes:, :]

q_dyn_sweep, growth_sweep, q_dyn_flutter = flutter_sweep_vs_dynamic_pressure(M_inf)

q_dyn_ND = (2 * q_dyn_flutter * a**3)/(((M_inf**2 - 2)/(M_inf**2 - 1)) * Dp)                            # Nondimensionalization based on Dowell/Adler Papers
print("Nondimensional Flutter Dynamic Pressure = ", q_dyn_ND)                                           

        # For Mach 2, Calculated Nondimensional Dynamic Pressure = 892.83 ; Dowell/Adler Nondimensional Dynamic Pressure = 866  -  3% error

# ======================== Plots =========================== #

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(q_dyn_sweep, growth_sweep, color='tab:blue', linewidth=2)
ax.axhline(0.0, color='k', linestyle='--', linewidth=1.0)
if q_dyn_flutter is not None:
    ax.axvline(q_dyn_flutter, color='tab:red', linestyle='--', label=f'Flutter: q_inf = {q_dyn_flutter:.3e} Pa')
    ax.legend()
ax.set_xlabel('Dynamic pressure q_inf [Pa]')
ax.set_ylabel('Largest eigenvalue real part')
ax.set_title('Flutter sweep vs dynamic pressure')
ax.grid(True)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
external_force = np.array([Aero_Modal_Force(solution.y[n_modes:, i], solution.y[:n_modes, i]) for i in range(solution.t.size)])
F1_ext = external_force[:, 0]

axes[0].plot(solution.t, F1_ext, color='tab:red', linewidth=2)
axes[0].set_ylabel('External Force Input')
axes[0].grid(True)
axes[0].set_title('External Force and Midpoint Displacement')

mode_index = np.arange(1, n_modes + 1, dtype=float)
midpoint_shape = np.sin(mode_index * np.pi / 2.0)
midpoint_disp = midpoint_shape @ qn

axes[1].plot(solution.t, midpoint_disp, color='tab:blue', linewidth=2)
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Midpoint displacement')
axes[1].grid(True)

plt.tight_layout()
plt.show()

q_final = solution.y[n_modes:, -1]
w_panel = phi.T @ q_final

plt.figure(figsize=(8, 5))
plt.plot(x, w_panel, color='tab:green', linewidth=2)
plt.xlabel('x [m]')
plt.ylabel('w displacement [m]')
plt.title('Final panel shape')
plt.grid(True)
plt.tight_layout()
plt.show()