
import pandas as pd
import sys
from dwave.system import LeapHybridSampler
from math import log2, floor
import dimod


def build_knapsack_bqm(costs, weights, weight_capacity):

    # Initialize BQM 
    bqm = dimod.AdjVectorBQM(dimod.Vartype.BINARY)
    lagrange = max(costs)

    x_size = len(costs)
    M = floor(log2(weight_capacity))
    num_slack_variables = M + 1

    y = [2**n for n in range(M)]
    y.append(weight_capacity + 1 - 2**M)

    for k in range(x_size):
        bqm.set_linear('x' + str(k), lagrange * (weights[k]**2) - costs[k])

    for i in range(x_size):
        for j in range(i + 1, x_size):
            key = ('x' + str(i), 'x' + str(j))
            bqm.quadratic[key] = 2 * lagrange * weights[i] * weights[j]

    for k in range(num_slack_variables):
        bqm.set_linear('y' + str(k), lagrange * (y[k]**2))

    for i in range(num_slack_variables):
        for j in range(i + 1, num_slack_variables):
            key = ('y' + str(i), 'y' + str(j))
            bqm.quadratic[key] = 2 * lagrange * y[i] * y[j]

    for i in range(x_size):
        for j in range(num_slack_variables):
            key = ('x' + str(i), 'y' + str(j))
            bqm.quadratic[key] = -2 * lagrange * weights[i] * y[j]

    return bqm

def solve_knapsack(costs, weights, weight_capacity, sampler=None):
    
    bqm = build_knapsack_bqm(costs, weights, weight_capacity)

    if sampler is None:
        sampler = LeapHybridSampler()

    sampleset = sampler.sample(bqm, label='Example - Knapsack')
    sample = sampleset.first.sample
    energy = sampleset.first.energy


    selected_item_indices = []
    for varname, value in sample.items():
        if value and varname.startswith('x'):
            selected_item_indices.append(int(varname[1:]))

    return sorted(selected_item_indices), energy


if __name__ == '__main__':

    data_file_name = sys.argv[1] if len(sys.argv) > 1 else "data/large.csv"
    weight_capacity = float(sys.argv[2]) if len(sys.argv) > 2 else 70

    #input data
    df = pd.read_csv(data_file_name, names=['cost', 'weight'])

    selected_item_indices, energy = solve_knapsack(df['cost'], df['weight'], weight_capacity)
    selected_weights = list(df.loc[selected_item_indices,'weight'])
    selected_costs = list(df.loc[selected_item_indices,'cost'])

    print("Found solution at energy {}".format(energy))
    print("Selected item numbers (0-indexed):", selected_item_indices)
    print("Selected item weights: {}, total = {}".format(selected_weights, sum(selected_weights)))
    print("Selected item costs: {}, total = {}".format(selected_costs, sum(selected_costs)))
