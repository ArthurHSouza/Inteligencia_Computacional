import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from poliastro.bodies import Earth
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit
from astropy import units as u

# Configurações Iniciais do Artigo [cite: 116, 123]
MU = Earth.k
R_EARTH = 6378.137 * u.km
ORB_I = Orbit.circular(Earth, alt=400*u.km) # LEO a 400km
R_I = ORB_I.a

def fitness_function(rho, r_f):
    """Calcula Delta-V total. rho = rb / rf [cite: 151, 158]"""
    if rho < 1.0:
        return 1e6, # Penalidade conforme seção 4.3 [cite: 161]
    
    r_b = rho * r_f
    try:
        # Utiliza a função bi-elíptica do Poliastro [cite: 147]
        man = Maneuver.bielliptic(ORB_I, r_b, r_f)
        return man.get_total_cost().to(u.m/u.s).value,
    except:
        return 1e6,

def run_optimization(scenario_name, r_f, rho_max):
    print(f"\n--- {scenario_name} ---")
    
    # Reset do framework DEAP para permitir múltiplas execuções
    if hasattr(creator, "FitnessMin"): del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    # Gene real-valorado conforme o artigo [cite: 151]
    toolbox.register("attr_rho", np.random.uniform, 1.0, rho_max)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_rho, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", lambda ind: fitness_function(ind[0], r_f))
    toolbox.register("mate", tools.cxBlend, alpha=0.5) # Crossover SBX [cite: 166]
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2) # Mutação [cite: 167]
    toolbox.register("select", tools.selTournament, tournsize=3) # Torneio [cite: 165]

    pop = toolbox.population(n=40) # População de 40 [cite: 163]
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    # Executa por 30 gerações [cite: 164]
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=30, stats=stats, verbose=False)
    
    best = tools.selBest(pop, 1)[0]
    
    # Comparação com Hohmann Analítico [cite: 205]
    h_man = Maneuver.hohmann(ORB_I, r_f)
    dv_hohmann = h_man.get_total_cost().to(u.m/u.s).value
    
    print(f"Melhor rho encontrado: {best[0]:.4f}")
    print(f"Delta-V GA: {best.fitness.values[0]:.2f} m/s")
    print(f"Delta-V Hohmann: {dv_hohmann:.2f} m/s")
    print(f"Diferença (Economia): {dv_hohmann - best.fitness.values[0]:.2f} m/s")
    
    return log.select("min")

# Execução dos dois cenários do artigo
# Cenário 1: LEO para GEO (rf/ri ≈ 6.2) [cite: 179, 180]
log_geo = run_optimization("Cenário 1: LEO para GEO", 42164 * u.km, 40.0)

# Cenário 2: LEO para Órbita Distante (rf/ri = 20) [cite: 191]
r_f_far = 20 * R_I
log_far = run_optimization("Cenário 2: LEO para Órbita Distante", r_f_far, 1000.0)

# Gerar Gráfico de Convergência
plt.plot(log_geo, label="Cenário 1 (GEO)")
plt.plot(log_far, label="Cenário 2 (Distante)")
plt.xlabel("Geração")
plt.ylabel("Delta-V Mínimo (m/s)")
plt.title("Convergência do Algoritmo Genético")
plt.legend()
plt.grid(True)
plt.show()