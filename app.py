import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Diseño Mecánico - Teorías de Falla",
    page_icon="⚙️",
    layout="wide",
)

st.title("Análisis Estático y de Fatiga")
st.markdown(
    "Basado en las teorías de falla estática (Tresca/von Mises) y carga fluctuante (Goodman/Gerber/Soderberg)."
)

# --- PANEL LATERAL: ENTRADAS ---
st.sidebar.header("1. Geometría y Apoyos (mm)")
L = st.sidebar.number_input("Longitud total L", value=500.0)
pos_a1 = st.sidebar.number_input("Posición Apoyo 1", value=0.0)
pos_a2 = st.sidebar.number_input("Posición Apoyo 2", value=500.0)

st.sidebar.header("2. Cargas Transversales (N)")
num_cargas = st.sidebar.slider("Número de fuerzas puntuales", 1, 3, 1)
cargas = []
for i in range(num_cargas):
    f_val = st.sidebar.number_input(f"Fuerza F{i+1} (- hacia abajo)", value=-1500.0, key=f"f_{i}")
    pos_val = st.sidebar.number_input(f"Posición x{i+1}", value=250.0, key=f"p_{i}")
    cargas.append({"pos": pos_val, "F": f_val})

st.sidebar.header("3. Momentos de Torsión (N·m)")
Ta = st.sidebar.number_input("Torsor Alternante Ta", value=80.0)
Tm = st.sidebar.number_input("Torsor Medio Tm", value=80.0)

st.sidebar.header("4. Material y Concentradores")
Sut = st.sidebar.number_input("Esfuerzo Último Sut (MPa)", value=700.0)
Sy = st.sidebar.number_input("Esfuerzo Fluencia Sy (MPa)", value=550.0)
Kf = st.sidebar.number_input("Kf (Flexión)", value=1.6)
Kfs = st.sidebar.number_input("Kfs (Torsión)", value=1.4)
n_objetivo = st.sidebar.number_input("Factor de Seguridad Objetivo (n)", value=2.0)

st.sidebar.header("5. Factores de Marin")
Cc = st.sidebar.number_input("Carga (Cc)", value=1.0)
Cd = st.sidebar.number_input("Temperatura (Cd)", value=1.0)
Ce = st.sidebar.number_input("Confiabilidad (Ce)", value=0.814)
Cf = st.sidebar.number_input("Efectos varios (Cf)", value=1.0)

# --- CÁLCULO ESTÁTICO DE REACCIONES Y DIAGRAMAS ---
dist_apoyos = pos_a2 - pos_a1
if dist_apoyos != 0:
    suma_mom_ext = sum(c["F"] * (c["pos"] - pos_a1) for c in cargas)
    R2 = -suma_mom_ext / dist_apoyos
    R1 = -sum(c["F"] for c in cargas) - R2
else:
    R1, R2 = 0.0, 0.0

x_vals = np.linspace(0, L, int(L) + 1)
V_vals, M_vals = np.zeros_like(x_vals), np.zeros_like(x_vals)

for idx, xi in enumerate(x_vals):
    v_acc, m_acc = 0, 0
    if xi >= pos_a1:
        v_acc += R1
        m_acc += R1 * (xi - pos_a1)
    for c in cargas:
        if xi >= c["pos"]:
            v_acc += c["F"]
            m_acc += c["F"] * (xi - c["pos"])
    if xi >= pos_a2:
        v_acc += R2
        m_acc += R2 * (xi - pos_a2)
    V_vals[idx], M_vals[idx] = v_acc, m_acc

M_max_Nmm = np.max(np.abs(M_vals))

# --- GRÁFICAS ---
st.subheader("📈 Diagramas de Carga")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax1.plot(x_vals, V_vals, color="tab:blue")
ax1.fill_between(x_vals, 0, V_vals, color="tab:blue", alpha=0.1)
ax1.set_ylabel("Cortante V (N)")
ax1.grid(True, linestyle="--")

ax2.plot(x_vals, M_vals / 1000.0, color="tab:orange")
ax2.fill_between(x_vals, 0, M_vals / 1000.0, color="tab:orange", alpha=0.1)
ax2.set_ylabel("Momento M (N·m)")
ax2.set_xlabel("Posición x (mm)")
ax2.grid(True, linestyle="--")
st.pyplot(fig)


# --- TEORÍAS DE FALLA (FATIGA Y ESTÁTICA) ---
def evaluar_diametro(d_test, M_max_Nmm):
    # 1. Factores de Marin
    Se_prime = 0.5 * Sut if Sut <= 1400 else 700.0
    
    # Factor de tamaño (Cb)
    if d_test <= 51:
        Cb = 1.24 * (d_test ** -0.107)
    else:
        Cb = 1.51 * (d_test ** -0.157)
    Cb = min(1.0, max(0.6, Cb))
    
    # Factor de superficie maquinado (Ca)
    Ca = min(1.0, 4.51 * (Sut ** -0.265))
    
    # Límite corregido
    Se = Ca * Cb * Cc * Cd * Ce * Cf * Se_prime # Cc=1, Cd=1, Ce(99%)=0.814, Cf=1

    # 2. Esfuerzos Equivalentes (considerando carga fluctuante)
    pi = np.pi
    # Flexión completamente invertida (Mm = 0)
    sigma_a = (32.0 * Kf * M_max_Nmm) / (pi * (d_test ** 3))
    sigma_m = 0.0 
    
    # Torsión fluctuante
    tau_a = (16.0 * Kfs * (Ta * 1000.0)) / (pi * (d_test ** 3))
    tau_m = (16.0 * Kfs * (Tm * 1000.0)) / (pi * (d_test ** 3))

    sigma_a_e = np.sqrt(sigma_a**2 + 3 * tau_a**2)
    sigma_m_e = np.sqrt(sigma_m**2 + 3 * tau_m**2)

    # 3. Factor de Seguridad a Fatiga (Goodman)
    inv_nf = (sigma_a_e / Se) + (sigma_m_e / Sut)
    nf = 1.0 / inv_nf if inv_nf > 0 else 999.0

    # 4. Factor de Seguridad Estático (Tresca y von Mises)
    sigma_max = sigma_a + sigma_m
    tau_max = tau_a + tau_m
    
    # Tresca (Cortante Máximo)
    tau_eq_tresca = np.sqrt((sigma_max / 2.0)**2 + tau_max**2)
    n_tresca = Sy / (2.0 * tau_eq_tresca) if tau_eq_tresca > 0 else 999.0
    
    # von Mises (Energía de Distorsión)
    sigma_eq_vm = np.sqrt(sigma_max**2 + 3 * tau_max**2)
    n_vm = Sy / sigma_eq_vm if sigma_eq_vm > 0 else 999.0

    return nf, n_tresca, n_vm, Se, Ca, Cb

# Búsqueda iterativa del diámetro
diametros = np.linspace(5, 150, 290)
d_fatiga, d_tresca, d_vm = None, None, None

for dt in diametros:
    nf_t, n_tresca_t, n_vm_t, _, _, _ = evaluar_diametro(dt, M_max_Nmm)
    if nf_t >= n_objetivo and d_fatiga is None: d_fatiga = dt
    if n_tresca_t >= n_objetivo and d_tresca is None: d_tresca = dt
    if n_vm_t >= n_objetivo and d_vm is None: d_vm = dt

# --- RESULTADOS ---
st.subheader("🔍 Diámetro Mínimo Requerido")
c1, c2, c3 = st.columns(3)
c1.success(f"**Fatiga (Goodman):** \n `{d_fatiga:.2f} mm`" if d_fatiga else "Fuera de rango")
c2.info(f"**Estática (Tresca):** \n `{d_tresca:.2f} mm`" if d_tresca else "Fuera de rango")
c3.warning(f"**Estática (von Mises):** \n `{d_vm:.2f} mm`" if d_vm else "Fuera de rango")

st.markdown("---")
st.subheader("🧮 Evaluación de Diámetro Específico")
d_user = st.slider("Diámetro de prueba (mm)", 10.0, 100.0, 35.0, 1.0)
nf_u, nt_u, nvm_u, Se_u, Ca_u, Cb_u = evaluar_diametro(d_user, M_max_Nmm)

m1, m2, m3 = st.columns(3)
m1.metric("F.S. Fatiga (Goodman)", f"{nf_u:.2f}")
m2.metric("F.S. Estático (Tresca)", f"{nt_u:.2f}")
m3.metric("F.S. Estático (von Mises)", f"{nvm_u:.2f}")
st.caption(f"Detalles Marin: Límite Corregido $S_e$ = `{Se_u:.2f} MPa` | $C_a$ = `{Ca_u:.3f}` | $C_b$ = `{Cb_u:.3f}`")
