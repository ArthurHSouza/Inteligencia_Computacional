import numpy
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

def fitness_function(rho, r_f, rho_max):
    r_b = rho * r_f

    """Calcula Delta-V total. rho = rb / rf [cite: 151, 158]"""
    if rho < 1.0 or rho > rho_max:
        return 1e6, # Penalidade conforme seção 4.3 [cite: 161]

    try:
        # Utiliza a função bi-elíptica do Poliastro [cite: 147]
        man = Maneuver.bielliptic(ORB_I, r_b, r_f)
        return man.get_total_cost().to(u.m/u.s).value,
    except:
        return 1e6,

def run_optimization(r_f, rho_max, initial_population=40, gen=30, tournsize=3, mprob=0.2):
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
    
    toolbox.register("evaluate", lambda ind: fitness_function(ind[0], r_f, rho_max))
    toolbox.register("mate", tools.cxBlend, alpha=0.5) 
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)

    pop = toolbox.population(n=initial_population) 
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    # Executa o algoritmo [cite: 164]
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=mprob, ngen=gen, stats=stats, verbose=False)
    
    best = tools.selBest(pop, 1)[0]

    r_b = best[0] * r_f

    # Retorna apenas a lista de mínimos para plotagem
    return r_b.value, best.fitness.values[0]

# --- EXECUÇÃO ---
ini_pop = 40
gen = 30
tournsize = 3
mprob = 0.9
r_f_far = 20 * R_I

medias_geo_rb = numpy.array([])
medias_geo_deltav = numpy.array([])


for i in range(0, 200):
    # Cenário 2: LEO para Órbita Distante (10x R_I no seu código)
    log_geo = run_optimization("Cenário 2: LEO para Órbita Distante", r_f_far, 1000.0,
                               initial_population=ini_pop, gen=gen, tournsize=tournsize)
    # Cenário 1: LEO para GEO [cite: 179]
    #log_geo = run_optimization("Cenário 1: LEO para GEO", 42164 * u.km, 40.0,
    #                       initial_population=ini_pop, gen=gen, tournsize=tournsize, mprob=mprob)
    medias_geo_rb = numpy.append(medias_geo_rb, log_geo[0])
    medias_geo_deltav = numpy.append(medias_geo_deltav, log_geo[1])

ga_man = Maneuver.bielliptic(ORB_I, medias_geo_rb.mean() * u.m, r_f_far)

print(f"Delta-V médio: {medias_geo_deltav.mean():.2f} m/s")
print(f"rb médio: {medias_geo_rb.mean()/1000:.2f} km")
print(f"Duração a partir do rb: {ga_man.get_total_time().to(u.day).value:.2f} dias")
