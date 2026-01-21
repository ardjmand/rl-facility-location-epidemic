from funcs import *
from params import *

# obj = EpidemicEnvironment(randomize=False)
obj = EpidemicEnvironment.load("../data/saved_problems/test.pt", device=device)
# obj.step(num_dt=3000, verbose=True, visualize=True, fit_mean_field=True, save_visualization=False)
obj.visualize_network(legend=True, save=False)

print("Done")