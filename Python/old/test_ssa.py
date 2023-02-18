

import numpy as np
import scipy.stats as st

#import gillespy2

# code from
# http://be150.caltech.edu/2019/handouts/12_stochastic_simulation_all_code.html

np.random.seed(777)


def simulate_coinflips(n, p, size=1):
    """
    Simulate n_samples sets of n coin flips with prob. p of heads.
    """
    n_heads = np.empty(size, dtype=int)
    for i in range(size):
        n_heads[i] = np.sum(np.random.random(size=n) < p)
    return n_heads



#class slm(gillespy2.Model):
#    def __init__(self, parameter_values=None):
#        # First call the gillespy2.Model initializer.
#        gillespy2.Model.__init__(self, name='Stochastic logistic model')

#        # Define parameters for the rates of creation and dissociation.
#        b = gillespy2.Parameter(name='b', expression=0.002)
#        d = gillespy2.Parameter(name='d', expression=0.001)
#        K = gillespy2.Parameter(name='K', expression=10000)
#        self.add_parameter([b, d, K])

#        # Define variables for the molecular species representing M and D.
#        n = gillespy2.Species(name='population size', initial_value=3000)
#        self.add_species([n])

#        # The list of reactants and products for a Reaction object are each a
#        # Python dictionary in which the dictionary keys are Species objects
#        # and the values are stoichiometries of the species in the reaction.
#        print(b*n)
#        birth_event = gillespy2.Reaction(name="birth", rate=b*n, reactants={n:1}, products={n:2})
#        death_event = gillespy2.Reaction(name="death", rate=((d + (((b-d)*n)/K))*n), reactants={n:1}, products={n:0})
#        self.add_reaction([birth_event, death_event])

#        # Set the timespan for the simulation.
#        self.timespan(np.linspace(0, 100, 101))

#model = slm()
#results = model.run(number_of_trajectories=1)



# Column 0 is change in m, column 1 is change in p
simple_update = np.array([[1],   # birth
                          [-1]],  # death]
                          dtype=np.int)



def simple_propensity(propensities, population_size, t, b, d, K):
    """Updates an array of propensities given a set of parameters
    and an array of populations.
    """
    # Unpack population
    n = population_size

    # Update propensities
    propensities[0] = b*n                               # birth
    propensities[1] = ((d + (((b-d)*n)/K))*n)           # death


def sample_discrete_scipy(probs):
    """Randomly sample an index with probability given by probs."""
    return st.rv_discrete(values=(range(len(probs)), probs)).rvs()



def sample_discrete(probs):
    """Randomly sample an index with probability given by probs."""
    # Generate random number
    q = np.random.rand()

    # Find index
    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return i - 1


def gillespie_draw(propensity_func, propensities, population, t, args=()):
    """
    Draws a reaction and the time it took to do that reaction.

    Parameters
    ----------
    propensity_func : function
        Function with call signature propensity_func(population, t, *args)
        used for computing propensities. This function must return
        an array of propensities.
    population : ndarray
        Current population of particles
    t : float
        Value of the current time.
    args : tuple, default ()
        Arguments to be passed to `propensity_func`.

    Returns
    -------
    rxn : int
        Index of reaction that occured.
    time : float
        Time it took for the reaction to occur.
    """
    # Compute propensities
    propensity_func(propensities, population, t, *args)

    # Sum of propensities
    props_sum = propensities.sum()

    # Compute next time
    time = np.random.exponential(1.0 / props_sum)

    # Compute discrete probabilities of each reaction
    rxn_probs = propensities / props_sum

    # Draw reaction from this distribution
    rxn = sample_discrete(rxn_probs)

    return rxn, time



def gillespie_ssa(propensity_func, update, population_0, time_points, args=()):
    """
    Uses the Gillespie stochastic simulation algorithm to sample
    from probability distribution of particle counts over time.

    Parameters
    ----------
    propensity_func : function
        Function of the form f(params, t, population) that takes the current
        population of particle counts and return an array of propensities
        for each reaction.
    update : ndarray, shape (num_reactions, num_chemical_species)
        Entry i, j gives the change in particle counts of species j
        for chemical reaction i.
    population_0 : array_like, shape (num_chemical_species)
        Array of initial populations of all chemical species.
    time_points : array_like, shape (num_time_points,)
        Array of points in time for which to sample the probability
        distribution.
    args : tuple, default ()
        The set of parameters to be passed to propensity_func.

    Returns
    -------
    sample : ndarray, shape (num_time_points, num_chemical_species)
        Entry i, j is the count of chemical species j at time
        time_points[i].
    """

    # Initialize output
    pop_out = np.empty((len(time_points), update.shape[1]), dtype=np.int)
    # Initialize and perform simulation
    i_time = 1
    i = 0
    t = time_points[0]
    population = population_0.copy()
    pop_out[0,:] = population
    propensities = np.zeros(update.shape[0])
    while i < len(time_points):
        while t < time_points[i_time]:
            # draw the event and time step
            event, dt = gillespie_draw(propensity_func, propensities, population, t, args)

            # Update the population
            population_previous = population.copy()
            population += update[event,:]

            # Increment time
            t += dt

        # Update the index
        i = np.searchsorted(time_points > t, True)

        # Update the population
        pop_out[i_time:min(i,len(time_points))] = population_previous

        # Increment index
        i_time = i

    return pop_out


# Specify parameters for calculation
args = (0.1, 0.0005, 1000)
#b = 0.1
#d = 0.005
#K = 1000
time_points = np.linspace(0, 500, 101)
population_0 = np.array([100], dtype=int)
size = 100


samples = np.empty((size, len(time_points), 1), dtype=int)


# Run the calculations
#for i in range(size):
print(gillespie_ssa(simple_propensity, simple_update,
                            population_0, time_points, args=args))
#samples[i,:,:] = gillespie_ssa(simple_propensity, simple_update,
#                            population_0, time_points, args=args)

#    #samples[i,:,:] = gillespie_ssa(simple_propensity, population_0, b, d, K)

#print(samples)
