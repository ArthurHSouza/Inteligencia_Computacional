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

def run_optimization(scenario_name, r_f, rho_max, initial_population=40, gen=30, tournsize=3, mprob=0.2):
    print(f"\n--- {scenario_name} ---")
    
    # Reset do framework DEAP
    if hasattr(creator, "FitnessMin"): del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    # Gene real-valorado [cite: 151]
    toolbox.register("attr_rho", np.random.uniform, 1.0, rho_max)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_rho, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", lambda ind: fitness_function(ind[0], r_f))
    toolbox.register("mate", tools.cxBlend, alpha=0.5) 
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=tournsize) 

    pop = toolbox.population(n=initial_population) 
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    # Executa o algoritmo [cite: 164]
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=mprob, ngen=gen, stats=stats, verbose=False)
    
    best = tools.selBest(pop, 1)[0]
    
    # Comparação com Hohmann 
    h_man = Maneuver.hohmann(ORB_I, r_f)
    dv_hohmann = h_man.get_total_cost().to(u.m/u.s).value
    
    print(f"Melhor rho encontrado: {best[0]:.4f}")
    print(f"Delta-V GA: {best.fitness.values[0]:.2f} m/s")
    print(f"Delta-V Hohmann: {dv_hohmann:.2f} m/s")
    print(f"Diferença (Economia): {dv_hohmann - best.fitness.values[0]:.2f} m/s")
    
    # Retorna apenas a lista de mínimos para plotagem
    return log.select("min")

# --- EXECUÇÃO ---
ini_pop = 40
gen = 30
tournsize = 3

# Cenário 1: LEO para GEO [cite: 179]
log_geo = run_optimization("Cenário 1: LEO para GEO", 42164 * u.km, 40.0, 
                           initial_population=ini_pop, gen=gen, tournsize=tournsize)

# Cenário 2: LEO para Órbita Distante (10x R_I no seu código)
r_f_far = 20 * R_I
log_far = run_optimization("Cenário 2: LEO para Órbita Distante", r_f_far, 1000.0, 
                           initial_population=ini_pop, gen=gen, tournsize=tournsize)

# --- CÁLCULO DAS REFERÊNCIAS DE HOHMANN ---
h_man1 = Maneuver.hohmann(ORB_I, 42164 * u.km)
dv_hohmann1 = h_man1.get_total_cost().to(u.m/u.s).value

h_man2 = Maneuver.hohmann(ORB_I, r_f_far)
dv_hohmann2 = h_man2.get_total_cost().to(u.m/u.s).value

# --- PLOTAGEM CORRIGIDA ---
plt.figure(figsize=(10, 6))

# Plotar a evolução do GA (Curvas de Convergência)
# Adicionamos uma sequência de 0 a gen para o eixo X explicitamente
generations = range(len(log_geo))
plt.plot(generations, log_geo, label="GA - Cenário 1 (GEO)", color='blue', marker='o', markersize=3)
plt.plot(generations, log_far, label="GA - Cenário 2 (Distante)", color='green', marker='x', markersize=3)

# Plotar as referências de Hohmann (Linhas Horizontais)
# axhline desenha uma linha em todo o eixo X na altura Y especificada
plt.axhline(y=dv_hohmann1, color='blue', linestyle='--', alpha=0.6, label=f"Ref. Hohmann GEO ({dv_hohmann1:.1f} m/s)")
plt.axhline(y=dv_hohmann2, color='green', linestyle='--', alpha=0.6, label=f"Ref. Hohmann Distante ({dv_hohmann2:.1f} m/s)")

plt.xlabel("Geração")
plt.ylabel("Delta-V Mínimo (m/s)")
plt.title("Convergência do Algoritmo Genético vs Referência Hohmann")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Opcional: Ajustar limites para ver melhor a convergência se os valores forem muito distantes
# plt.ylim(min(log_far + log_geo) - 100, max(log_far + log_geo) + 100)

plt.show()