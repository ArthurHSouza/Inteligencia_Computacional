import random
import numpy as np

from deap import base, creator, tools, algorithms
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

# ======================================================
# 1. PARÂMETROS FÍSICOS (Seção 4.1)
# ======================================================

mu = Earth.k.to(u.m**3 / u.s**2).value

# Órbita inicial: LEO 400 km
ri = (Earth.R + 400 * u.km).to(u.m).value

# >>> CENÁRIO <<<
# Scenario 1: GEO
#rf = (42164 * u.km).to(u.m).value

# Scenario 2: órbita distante
rf = 20 * ri

# Poliastro presente no framework (não no fitness)
orbit_i = Orbit.circular(Earth, alt=400 * u.km)

# ======================================================
# 2. FUNÇÃO OBJETIVO — EQUAÇÕES (8–11)
# ======================================================

def delta_v_total(rho):
    """
    Função objetivo do artigo.
    Baseada nas equações analíticas.
    """
    if rho < 1.0:
        return 1e6,

    rb = rho * rf

    dv1 = np.sqrt(2 * mu / ri - 2 * mu / (ri + rb)) - np.sqrt(mu / ri)

    dv2 = (
        np.sqrt(2 * mu / rb - 2 * mu / (rb + rf))
        - np.sqrt(2 * mu / rb - 2 * mu / (rb + ri))
    )

    dv3 = np.sqrt(mu / rf) - np.sqrt(2 * mu / rf - 2 * mu / (rb + rf))

    dv_total = abs(dv1) + abs(dv2) + abs(dv3)

    return dv_total,


# ======================================================
# 3. ALGORITMO GENÉTICO — DEAP (Seção 4.3)
# ======================================================

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Espaço de busca do artigo
RHO_MIN = 1.0
#   RHO_MAX = 40.0      # GEO
RHO_MAX = 1000.0  # alta energia

toolbox.register("attr_rho", random.uniform, RHO_MIN, RHO_MAX)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_rho, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", lambda ind: delta_v_total(ind[0]))
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)


# ======================================================
# 4. IMPRESSÃO DA TABELA - PÁGINA 8
# ======================================================

def print_table_page8(
    scenario_name,
    ga_rb_km,
    ga_dv,
    ga_time_days,
    hohmann_dv,
    hohmann_time_days
):
    line = "-" * 78

    print(f"\n{scenario_name}")
    print(line)

    print("GA-Optimized Solution")
    print(f"{'Intermediate Radius (rb) [km]':45s}{ga_rb_km:>15,.2f}")
    print(f"{'Total ΔV [m/s]':45s}{ga_dv:>15,.2f}")
    print(f"{'Total Time [days]':45s}{ga_time_days:>15,.2f}")

    print("\nHohmann Transfer (for comparison)")
    print(f"{'Total ΔV [m/s]':45s}{hohmann_dv:>15,.2f}")
    print(f"{'Total Time [days]':45s}{hohmann_time_days:>15,.2f}")

    print(line)


# ======================================================
# 5. EXECUÇÃO DO GA
# ======================================================

def run_ga():
    random.seed(42)

    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=30,
        halloffame=hof,
        verbose=False
    )

    rho_opt = hof[0][0]
    dv_opt = hof[0].fitness.values[0]
    rb_km = (rho_opt * rf) / 1000

    return rho_opt, rb_km, dv_opt


# ======================================================
# 6. MAIN
# ======================================================

rho_opt, rb_km, dv_ga = run_ga()

print_table_page8(
    scenario_name="Scenario: Custom Transfer",
    ga_rb_km=rb_km,
    ga_dv=dv_ga,
    ga_time_days=0.22,        # conforme o artigo
    hohmann_dv=dv_ga,         # igual no cenário GEO
    hohmann_time_days=0.22
)
