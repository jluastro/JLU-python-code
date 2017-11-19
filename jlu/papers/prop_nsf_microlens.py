import numpy as np

def prob_bh_mass(mass, alpha, m_gap, m_cap):
    part1 = mass**{-alpha}
    part2 = np.heaviside(mass, m_gap)
    part3 = np.exp(-mass / m_cap)

    prob = part1 * part2 * part3

    return prob
    
def junk():

    
    norm_values = scipy.stats.norm.pdf(m_obs / mass, 1, m_err)
    prob_m_obs = prob_bh_mass(mass, alpha, m_gap, m_cap) * norm_values * (1.0 / mass)

    # Observable:
    # Total number of detected microlensing black holes.
    
