import numpy
from scipy.integrate import solve_ivp
from bfm.bfm50.bfm50_rate_eqns import reduced_bfm50_rate_eqns
import time
import datetime

def solution_time(conc, bfm_phys_vars, mult,):
    """ calculates the computation time for uncoupled bfm integration (RK23)
    """

    integration_time = numpy.zeros(5)
    # Time span for integration
    t_span = [0, 86400*365*10]
    
    for i in range(0,5):

        # Record integration start time
        start = time.process_time()

        # Integrate model
        solution_full_model = solve_ivp(lambda time, conc: reduced_bfm50_rate_eqns(time, conc, bfm_phys_vars, mult, True, 150, True), t_span, conc.ravel(), method='RK23')


        # Record integration end time
        end = time.process_time()

        # Time of integration
        integration_time[i] = end - start
        print(integration_time[i])

    mean = numpy.mean(integration_time)
    avg_integration_time = str(datetime.timedelta(seconds=mean))

    return avg_integration_time


def extract_dict(dictionary):

    data = list(dictionary.values())
    values = numpy.array(data)

    return values


def fill_dict(dictionary,array,species_names):

    for i, sp in enumerate(species_names):
        dictionary[sp] = array[i]
    
    return dictionary

