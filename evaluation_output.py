from utils import (load_problem_data,
                   load_solution)
from evaluation import evaluation_function
from seeds import known_seeds
import statistics

def evaluate(path=None):
    seeds = known_seeds('test')
    if path==None:
        path = './'
    scores = []
    for seed in seeds:
        print(f"SEED: #{seed}")
        # LOAD SOLUTION
        solution = load_solution(f'{path}output/{seed}.json')

        # LOAD PROBLEM DATA
        demand, datacenters, servers, selling_prices = load_problem_data(path=f'{path}data')

        # EVALUATE THE SOLUTION
        score = evaluation_function(solution,
                                    demand,
                                    datacenters,
                                    servers,
                                    selling_prices,
                                    seed=seed,
                                    verbose=1)
        scores.append(score)
        print(f'Solution score: {score}')

    print(f'Average solution score: {statistics.mean(scores)}')
    return statistics.mean(scores)


if __name__ == "__main__":
    evaluate()