from params import *


class EpidemicEnvironment(MessagePassing):
    def __init__(self,
                 randomize=False,
                 env_params=env_params,
                 env_params_range=env_params_range,
                 compartments=compartments,
                 compartments_abbr=compartments_abbr,
                 compartments_colors=compartments_colors,
                 init_compartments=init_compartments):
        super(EpidemicEnvironment, self).__init__(aggr='add')

        self.randomize = randomize
        self.env_params = copy.deepcopy(env_params)
        self.env_params_range = copy.deepcopy(env_params_range)
        self.compartments = copy.deepcopy(compartments)
        self.compartments_colors = compartments_colors
        self.compartments_abbr = copy.deepcopy(compartments_abbr)
        self.init_compartments = copy.deepcopy(init_compartments)

        if self.randomize:
            self._randomize_params()

        # Extract parameters after potential randomization
        self.N = self.env_params["N"]
        self.M = self.env_params["M"]
        self.beta_1 = torch.tensor(self.env_params["beta_1"], device=device, dtype=torch.float32)
        self.beta_2 = torch.tensor(self.env_params["beta_2"], device=device, dtype=torch.float32)
        self.delta = torch.tensor(self.env_params["delta"], device=device, dtype=torch.float32)
        self.omega = torch.tensor(self.env_params["omega"], device=device, dtype=torch.float32)
        self.v_min = torch.tensor(self.env_params["v_min"], device=device, dtype=torch.float32)
        self.v_max = torch.tensor(self.env_params["v_max"], device=device, dtype=torch.float32)
        self.alpha = torch.tensor(self.env_params["alpha"], device=device, dtype=torch.float32)
        self.f_plus = torch.tensor(self.env_params["f_plus"], device=device, dtype=torch.float32)
        self.f_minus = torch.tensor(self.env_params["f_minus"], device=device, dtype=torch.float32)
        self.C_O = torch.tensor(self.env_params["C_O"], device=device, dtype=torch.float32)
        self.C_I = torch.tensor(self.env_params["C_I"], device=device, dtype=torch.float32)
        self.C_V = torch.tensor(self.env_params["C_V"], device=device, dtype=torch.float32)
        self.network_type = self.env_params["network_type"]
        self.avg_deg = self.env_params["avg_deg"]
        self.pct_open_fac = self.env_params["pct_open_fac"]
        self.dt = self.env_params["dt"]

        self.data = self.generate_graph()
        self.min_dists_to_open_facilities = self._compute_min_dists_to_open_facilities()

    def _randomize_params(self):
        for param, rng in self.env_params_range.items():
            if param == "network_type":
                # Randomize from a categorical list
                self.env_params["network_type"] = random.choice(rng)
            else:
                # rng should be a tuple (min_val, max_val) for numeric params
                if isinstance(rng, tuple) and len(rng) == 2:
                    if isinstance(rng[0], int) and isinstance(rng[1], int):
                        self.env_params[param] = random.randint(rng[0], rng[1])
                    else:
                        self.env_params[param] = random.uniform(rng[0], rng[1])
                else:
                    # If not a tuple, we assume it's categorical (for future expansions)
                    # This code won't run here since we handled network_type above.
                    pass

        # ensure beta_2 is always less than or equal to beta_1 by setting its upper bound equal to beta_1 value
        if self.env_params["beta_2"] > self.env_params["beta_1"]:
            self.env_params["beta_2"] = random.uniform(self.env_params_range["beta_2"][0], self.env_params["beta_1"])

        # ensure v_min is always less than or equal to v_min
        if self.env_params["v_min"] > self.env_params["v_max"]:
            self.env_params["v_min"] = random.uniform(self.env_params_range["v_min"][0], self.env_params["v_max"])

        # Randomize initial compartment percentages so they sum up to 1
        init_values = torch.rand(3)
        init_values = init_values / init_values.sum()

        self.init_compartments["S"] = float(init_values[0])
        self.init_compartments["I"] = float(init_values[1])
        self.init_compartments["V"] = float(init_values[2])

    def _compute_min_dists_to_open_facilities(self):
        """
        Computes and stores the minimum distance from each individual
        to the closest open facility, stored as a class attribute on GPU.
        """
        coords_ind = self.data['individual'].x[:, 3:]
        coords_fac = self.data['facility'].x[:, 1:]
        open_mask = self.data['facility'].x[:, 0] == 1

        if open_mask.any():
            open_coords = coords_fac[open_mask]
            dists = torch.cdist(coords_ind, open_coords)  # [N, num_open]
            min_dists_to_open_facilities = dists.min(dim=1).values  # [N]
        else:
            min_dists_to_open_facilities = torch.full((self.N,), float('inf'), device=coords_ind.device)

        return min_dists_to_open_facilities

    def generate_graph(self):

        # === 1. Randomly assign compartments (S, I, and V) to each individual node ===
        # Convert list of numpy arrays to single numpy array first to avoid slow tensor creation
        compartments_array = np.array(list(self.compartments.values()), dtype=np.float32)
        compartments_tensor = torch.from_numpy(compartments_array).to(device=device)
        pct_tensor = torch.tensor([
            self.init_compartments["S"],
            self.init_compartments["I"],
            self.init_compartments["V"]
        ], dtype=torch.float, device=device)
        num_rows = (pct_tensor * self.N).round().to(torch.int)

        # Adjust to ensure the total number of rows equals n
        diff = self.N - num_rows.sum()
        if diff != 0:
            # Find the index with the maximum percentage to adjust
            max_idx = torch.argmax(pct_tensor)
            num_rows[max_idx] += diff

        # Use repeat_interleave to create the rows based on the computed number of rows
        ind_state = compartments_tensor.repeat_interleave(num_rows, dim=0)
        ind_state = ind_state.to(device=device)

        # Randomly permute the rows
        indices = torch.randperm(ind_state.size(0), device=device)
        ind_state = ind_state[indices]

        # === 2. add coordinate of each individual to their features ===
        coord_i = torch.rand(self.N, 2, device=device)
        x_individual = torch.cat([ind_state, coord_i], dim=1)  # [N, 5]

        # === 3. create random vaccination facilities and set their status (open/closed) randomly ===
        coord_f = torch.rand(self.M, 2, device=device)
        fac_open_prob = torch.tensor([1 - self.pct_open_fac, self.pct_open_fac], dtype=torch.float, device=device)
        fac_open = torch.multinomial(fac_open_prob, self.M, replacement=True)
        fac_open = fac_open.to(device=device)
        fac_open = torch.reshape(fac_open, [self.M, 1])
        x_facility = torch.cat([fac_open, coord_f], dim=1)  # [M, 3]

        # === 4. add individual-individual edges ===
        dist = torch.cdist(coord_i, coord_i)  # [N_i, N_i]
        dist.fill_diagonal_(float('inf'))  # prevent self-loops

        sim = torch.exp(-dist / 0.05)  # connection weight ∝ similarity (closer = more similar)

        if self.network_type == 'renyi':
            # Normalize similarity to [0, 1]
            prob = sim / sim.max()

            # Scale probabilities to preserve expected average degree
            scale = (self.N * self.avg_deg) / (2 * prob.triu(diagonal=1).sum().clamp(min=1e-6))
            prob = prob * scale
            prob = prob.clamp(max=1.0)

            # Sample upper triangle
            upper_mask = torch.triu(torch.ones_like(prob), diagonal=1).bool()
            sampled = torch.bernoulli(prob).bool() & upper_mask

            # Symmetrize
            edge_mask = sampled | sampled.T

            # Extract edge index
            src, dst = edge_mask.nonzero(as_tuple=True)
            edge_index_ii = torch.stack([src, dst], dim=0)

        elif self.network_type == 'expo':
            # Normalize proximity probabilities row-wise
            prob = sim / (sim.sum(dim=1, keepdim=True) + 1e-9)

            # Exponentially distributed degrees via inverse transform
            lambda_param = 1.0 / self.avg_deg
            u = torch.rand(self.N, device=device)
            degrees = (-torch.log(1 - u) / lambda_param).floor().clamp(min=1, max=self.N - 1).int()

            max_k = degrees.max().item()
            samples = torch.multinomial(prob, num_samples=max_k, replacement=False)
            row_idx = torch.arange(self.N, device=device).view(-1, 1).expand(-1, max_k)
            mask = torch.arange(max_k, device=device).view(1, -1) < degrees.view(-1, 1)

            src_nodes = row_idx[mask]
            dst_nodes = samples[mask]
            valid = src_nodes != dst_nodes
            src_nodes = src_nodes[valid]
            dst_nodes = dst_nodes[valid]

            # Symmetrize edges by adding reverse pairs
            edge_index_ii = torch.cat([
                torch.stack([src_nodes, dst_nodes], dim=0),
                torch.stack([dst_nodes, src_nodes], dim=0)
            ], dim=1)

            # Optional: remove duplicate edges (undirected treatment)
            edge_index_ii = torch.unique(edge_index_ii, dim=1)

        else:
            raise ValueError("network_type must be 'renyi' or 'expo'")

        # === 5. Individual–Facility edges + distance ===
        dist_if = torch.cdist(coord_i, coord_f)
        src = torch.arange(self.N, device=device).repeat_interleave(self.M)
        dst = torch.arange(self.M, device=device).repeat(self.N)
        edge_index_if = torch.stack([src, dst], dim=0)
        edge_attr_if = dist_if.flatten().unsqueeze(1)

        # === 6. Facility–Facility edges + distance ===
        dist_ff = torch.cdist(coord_f, coord_f)  # shape: [M, M]
        dist_ff.fill_diagonal_(float('inf'))  # ignore self-loops
        # Flatten upper triangle (since it's symmetric and undirected)
        mask = torch.triu(torch.ones_like(dist_ff), diagonal=1).bool()
        src_ff, dst_ff = mask.nonzero(as_tuple=True)
        edge_index_ff = torch.stack([src_ff, dst_ff], dim=0)
        # Symmetrize: include (j, i) for each (i, j)
        edge_index_ff = torch.cat([edge_index_ff, edge_index_ff.flip(0)], dim=1)
        # Edge attributes: distances for the symmetric edges
        edge_attr_ff = dist_ff[src_ff, dst_ff]
        edge_attr_ff = torch.cat([edge_attr_ff, edge_attr_ff], dim=0).unsqueeze(1)  # shape: [2 * num_edges, 1]

        # === 7. Package HeteroData ===
        data = HeteroData()
        data['individual'].x = x_individual
        data['facility'].x = x_facility
        data['individual', 'interacts', 'individual'].edge_index = edge_index_ii
        data['individual', 'visits', 'facility'].edge_index = edge_index_if
        data['individual', 'visits', 'facility'].edge_attr = edge_attr_if
        # Reverse edge: facility → individual
        data['facility', 'visited_by', 'individual'].edge_index = edge_index_if.flip(0)
        data['facility', 'visited_by', 'individual'].edge_attr = edge_attr_if  # same distance
        data['facility', 'connects', 'facility'].edge_index = edge_index_ff
        data['facility', 'connects', 'facility'].edge_attr = edge_attr_ff

        return data

    def visualize_network(self, legend=True, save=True):

        # Extract data to CPU
        coords_ind = self.data['individual'].x[:, -2:].cpu().numpy()
        states = self.data['individual'].x[:, :3].argmax(dim=1).cpu().numpy()
        coords_fac = self.data['facility'].x[:, -2:].cpu().numpy()
        fac_status = self.data['facility'].x[:, 0].cpu().numpy()

        fig, ax = plt.subplots()

        # Plot contact network edges using LineCollection (much faster than loop)
        edge_index = self.data['individual', 'interacts', 'individual'].edge_index.cpu().numpy()
        edge_segments = [[coords_ind[i], coords_ind[j]] for i, j in zip(*edge_index)]
        ax.add_collection(LineCollection(edge_segments, colors='black', linewidths=1, alpha=0.6, zorder=1))

        # Plot individual → closest open facility edges
        open_mask = fac_status == 1
        if open_mask.any():
            coords_open = coords_fac[open_mask]
            dists = np.linalg.norm(coords_ind[:, None, :] - coords_open[None, :, :], axis=2)
            closest_coords = coords_open[dists.argmin(axis=1)]
            fac_segments = [[coords_ind[i], closest_coords[i]] for i in range(len(coords_ind))]
            ax.add_collection(LineCollection(fac_segments, colors='darkgrey', linestyles='--', linewidths=1, alpha=0.3, zorder=1))

        # Plot individuals by compartment
        for idx, (label, color) in enumerate(self.compartments_colors.items()):
            mask = states == idx
            if mask.any():
                ax.scatter(coords_ind[mask, 0], coords_ind[mask, 1], c=[color], s=80,
                           edgecolor='black', label=self.compartments_abbr[label], zorder=2)

        # Plot facilities
        if open_mask.any():
            ax.scatter(coords_fac[open_mask, 0], coords_fac[open_mask, 1],
                       c='orange', marker='^', s=100, label='Facility (Open)', zorder=2)
        if (~open_mask).any():
            ax.scatter(coords_fac[~open_mask, 0], coords_fac[~open_mask, 1],
                       c='grey', marker='^', s=100, label='Facility (Closed)', alpha=0.5, zorder=2)

        ax.axis('off')
        if legend:
            ax.legend(loc='upper left')
        plt.tight_layout()
        if save:
            filename = f"network_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf"
            plt.savefig(filename, format='pdf')
        plt.show()

    def compute_vaccination_rates(self):
        """
        Computes the S → V transition rate based on distance to the nearest open facility.
        """
        if not hasattr(self, 'min_dists_to_open_facilities'):
            raise RuntimeError("Minimum distances not computed. Call _compute_min_dists_to_open_facilities first.")

        SV_rate = self.v_min + (self.v_max - self.v_min) * torch.exp(-self.alpha * self.min_dists_to_open_facilities)
        return SV_rate

    def forward(self, x_individual, edge_index_ii, x_facility=None, edge_index_if=None):
        """

        :param x_individual: [N_ind, 5] -> [S, I, V, coord_x, coord_y]
        :param edge_index_ii: individual–individual edges
        :param x_facility: [open/closed, coord_x, coord_y]
        :param edge_index_if: individual–facility edges
        """
        S = x_individual[:, 0]
        I = x_individual[:, 1]
        V = x_individual[:, 2]
        num_nodes = x_individual.size(0)

        # Step 1: transition rates
        agg_neighbors = self.propagate(edge_index_ii, x=x_individual)
        infected_neighbors = agg_neighbors[:, 1]  # sum of 'I' values from neighbors

        SI_rate = self.beta_1 * infected_neighbors
        VI_rate = self.beta_2 * infected_neighbors
        IV_rate = self.delta.expand(num_nodes).to(x_individual.device)
        VS_rate = self.omega.expand(num_nodes).to(x_individual.device)
        SV_rate = self.compute_vaccination_rates()

        new_ind_state = x_individual.clone()

        # Step 2: Transition S → I or V
        total_rate_from_S = SI_rate + SV_rate
        pr_transition_from_S = 1 - torch.exp(-total_rate_from_S * self.dt)
        random_vals = torch.rand(num_nodes, device=x_individual.device)
        will_transition_from_S = (random_vals < pr_transition_from_S) & (S > 0)
        pr_SI = SI_rate / total_rate_from_S
        pr_SV = SV_rate / total_rate_from_S
        random_vals_transition = torch.rand(num_nodes, device=x_individual.device)
        transition_from_S_to_I = (will_transition_from_S & (random_vals_transition < pr_SI))
        transition_from_S_to_V = (will_transition_from_S & (random_vals_transition >= pr_SI))
        new_ind_state[transition_from_S_to_I, 1] = 1  # transition to compartment I
        new_ind_state[transition_from_S_to_V, 2] = 1  # transition to compartment V
        new_ind_state[transition_from_S_to_I | transition_from_S_to_V, 0] = 0

        # Step 3: Transition V → S or I
        total_rate_from_V = VI_rate + VS_rate
        pr_transition_from_V = 1 - torch.exp(-total_rate_from_V * self.dt)
        random_vals = torch.rand(num_nodes, device=x_individual.device)
        will_transition_from_V = (random_vals < pr_transition_from_V) & (V > 0)
        pr_VI = VI_rate / total_rate_from_V
        pr_VS = VS_rate / total_rate_from_V
        random_vals_transition = torch.rand(num_nodes, device=x_individual.device)
        transition_from_V_to_I = (will_transition_from_V & (random_vals_transition < pr_VI))
        transition_from_V_to_S = (will_transition_from_V & (random_vals_transition >= pr_VI))
        new_ind_state[transition_from_V_to_I, 1] = 1  # transition to compartment I
        new_ind_state[transition_from_V_to_S, 0] = 1  # transition to compartment S
        new_ind_state[transition_from_V_to_I | transition_from_V_to_S, 2] = 0

        # Step 4: Transition I → V
        total_rate_from_I = IV_rate.clone()
        pr_transition_from_I = 1 - torch.exp(-total_rate_from_I * self.dt)
        random_vals = torch.rand(num_nodes, device=x_individual.device)
        will_transition_from_I = (random_vals < pr_transition_from_I) & (I > 0)
        pr_IV = IV_rate / total_rate_from_I
        random_vals_transition = torch.rand(num_nodes, device=x_individual.device)
        transition_from_I_to_V = (will_transition_from_I & (random_vals_transition < pr_IV))
        new_ind_state[transition_from_I_to_V, 2] = 1  # transition to compartment V
        new_ind_state[transition_from_I_to_V, 1] = 0

        return new_ind_state

    def message(self, x_j):
        # x_j: Features of neighboring nodes
        return x_j

    def step(self, num_dt, fit_mean_field=False, verbose=True, visualize=True, save_visualization=True, return_details=False):
        """
        Run epidemic simulation for num_dt timesteps.

        Args:
            num_dt: Number of timesteps to simulate
            fit_mean_field: Whether to fit mean-field approximation
            verbose: Print progress
            visualize: Plot results
            save_visualization: Save plots to file
            return_details: If True, return detailed cost breakdown for RL (optimized GPU path)

        Returns:
            If return_details=False: (cost, comp_history)
            If return_details=True: (cost, comp_history, details_dict)
                details_dict contains: total_infections, total_vaccinations,
                individual_cost, facility_cost, current_infected
        """
        if fit_mean_field:
            # create a copy of the self to be used for mean-field approximation.
            # This copy is needed because the simulation of the step function changes the self.
            self_copy = copy.deepcopy(self)

        # Fast path for RL: skip history, accumulate on GPU
        if return_details and not verbose and not visualize:
            return self._step_fast(num_dt)

        # store compartment distributions over time
        comp_history = []
        comp_distribution = torch.mean(self.data["individual"].x, dim=0)
        comp_history.append(comp_distribution.cpu().detach().numpy())
        if verbose:
            print(f"{'Initial Compartments':<15} "
                  f"S={comp_distribution[0]:.3f},  "
                  f"I={comp_distribution[1]:.3f},  "
                  f"V={comp_distribution[2]:.3f}")

        # Accumulators on GPU (no .item() in loop)
        total_new_infected = torch.tensor(0, dtype=torch.long, device=device)
        total_new_vaccinated = torch.tensor(0, dtype=torch.long, device=device)

        # Cache edge indices (avoid repeated dict lookups)
        edge_index_ii = self.data['individual', 'interacts', 'individual'].edge_index
        edge_index_if = self.data['individual', 'visits', 'facility'].edge_index
        facility_x = self.data["facility"].x

        # Number of open facilities (constant during this step)
        num_open = (facility_x[:, 0] == 1).sum()

        for s in range(num_dt):
            # perform a forward pass to update states
            current_susceptible = self.data["individual"].x[:, 0]
            current_infected = self.data["individual"].x[:, 1]

            self.data["individual"].x = self.forward(
                self.data["individual"].x,
                edge_index_ii,
                facility_x,
                edge_index_if
            )

            new_infected = self.data["individual"].x[:, 1]
            new_vaccinated = self.data["individual"].x[:, 2]

            # Accumulate on GPU (no sync)
            total_new_infected += ((new_infected == 1) & (current_infected == 0)).sum()
            total_new_vaccinated += ((new_vaccinated == 1) & (current_susceptible == 1)).sum()

            # store the current compartment distribution
            comp_distribution = torch.mean(self.data["individual"].x, dim=0)
            comp_history.append(comp_distribution.cpu().detach().numpy())

            if verbose:
                if s % 100 == 0:
                    print(f"{'dt=' + str(s + 1):<15} "
                          f"S={comp_distribution[0]:.3f},  "
                          f"I={comp_distribution[1]:.3f},  "
                          f"V={comp_distribution[2]:.3f}")

        # Single GPU->CPU sync at the end
        num_infected = total_new_infected.item()
        num_vaccinated = total_new_vaccinated.item()
        num_open_val = num_open.item()

        # Compute costs (convert tensor params to Python floats)
        C_I = self.C_I.item()
        C_V = self.C_V.item()
        C_O = self.C_O.item()
        individual_cost = C_I * num_infected + C_V * num_vaccinated
        facility_cost = C_O * num_open_val * num_dt
        cost = individual_cost + facility_cost

        if verbose:
            print(f"{'Final State':<15} "
                  f"S={comp_distribution[0]:.3f},  "
                  f"I={comp_distribution[1]:.3f},  "
                  f"V={comp_distribution[2]:.3f}")

        if visualize:
            fig, ax = plt.subplots()
            t = [i * self.dt for i in range(num_dt + 1)]
            ax.plot(t, [s[0] for s in comp_history], linewidth=2, color=compartments_colors["S"], label="Susceptible (S)", alpha=0.7)
            ax.plot(t, [s[1] for s in comp_history], linewidth=2, color=compartments_colors["I"], label="Infected (I)", alpha=0.7)
            ax.plot(t, [s[2] for s in comp_history], linewidth=2, color=compartments_colors["V"], label="Vaccinated (V)", alpha=0.7)

            if fit_mean_field:
                S, I, V, mean_field_t = self_copy.mean_field(num_dt=num_dt, visualize=False)
                ax.plot(mean_field_t, S.mean(axis=1), linewidth=1, linestyle="--", color="black", label="Mean-Field")
                ax.plot(mean_field_t, I.mean(axis=1), linewidth=1, linestyle="--", color="black")
                ax.plot(mean_field_t, V.mean(axis=1), linewidth=1, linestyle="--", color="black")

            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Percentage of individuals in each Compartment", fontsize=12)
            plt.grid(linestyle="--", linewidth=1, alpha=0.5)
            plt.legend(loc="upper right")
            plt.tight_layout()
            if save_visualization:
                plt.savefig("comp_pct.pdf")
            plt.show()

            # Probability of compartments for each individual
            if fit_mean_field:
                fig, ax = plt.subplots()
                ax.plot(mean_field_t, S[:, 0], linewidth=1, color=self.compartments_colors["S"], alpha=0.4,
                        label="Susceptible (S)")
                ax.plot(mean_field_t, I[:, 0], linewidth=1, color=self.compartments_colors["I"], alpha=0.4,
                        label="Infected (I)")
                ax.plot(mean_field_t, V[:, 0], linewidth=1, color=self.compartments_colors["V"], alpha=0.4,
                        label="Vaccinated (V)")
                ax.plot(mean_field_t, S[:, 1:], linewidth=1, color=self.compartments_colors["S"], alpha=0.4)
                ax.plot(mean_field_t, I[:, 1:], linewidth=1, color=self.compartments_colors["I"], alpha=0.4)
                ax.plot(mean_field_t, V[:, 1:], linewidth=1, color=self.compartments_colors["V"], alpha=0.4)
                ax.set_xlabel("Time", fontsize=12)
                ax.set_ylabel("Probability of compartments for each individual", fontsize=12)
                plt.grid(linestyle="--", linewidth=1, alpha=0.5)
                plt.legend(loc="upper right")
                plt.tight_layout()
                if save_visualization:
                    plt.savefig("comp_pct_ind.pdf")
                plt.show()

        if return_details:
            current_infected = (self.data["individual"].x[:, 1] == 1).sum().item()
            details = {
                'total_infections': num_infected,
                'total_vaccinations': num_vaccinated,
                'individual_cost': individual_cost,
                'facility_cost': facility_cost,
                'current_infected': current_infected,
            }
            return cost, comp_history, details

        return cost, comp_history

    def _step_fast(self, num_dt):
        """
        Optimized simulation step for RL training.
        Minimizes GPU-CPU synchronization by accumulating on GPU tensors.
        No history tracking, no visualization - pure speed.

        Args:
            num_dt: Number of timesteps to simulate

        Returns:
            (cost, None, details_dict) - comp_history is None for speed
        """
        # Cache edge indices (avoid repeated dict lookups)
        edge_index_ii = self.data['individual', 'interacts', 'individual'].edge_index
        edge_index_if = self.data['individual', 'visits', 'facility'].edge_index
        facility_x = self.data["facility"].x

        # Accumulators on GPU
        total_new_infected = torch.zeros(1, dtype=torch.long, device=device)
        total_new_vaccinated = torch.zeros(1, dtype=torch.long, device=device)

        # Number of open facilities (constant during this step)
        num_open = (facility_x[:, 0] == 1).sum()

        for _ in range(num_dt):
            # Store current states for transition detection
            current_susceptible = self.data["individual"].x[:, 0]
            current_infected = self.data["individual"].x[:, 1]

            # Forward pass (GPU message passing)
            self.data["individual"].x = self.forward(
                self.data["individual"].x,
                edge_index_ii,
                facility_x,
                edge_index_if
            )

            new_infected = self.data["individual"].x[:, 1]
            new_vaccinated = self.data["individual"].x[:, 2]

            # Accumulate on GPU (no sync)
            total_new_infected += ((new_infected == 1) & (current_infected == 0)).sum()
            total_new_vaccinated += ((new_vaccinated == 1) & (current_susceptible == 1)).sum()

        # Single GPU->CPU sync at the end
        num_infected = total_new_infected.item()
        num_vaccinated = total_new_vaccinated.item()
        num_open_val = num_open.item()
        current_infected = (self.data["individual"].x[:, 1] == 1).sum().item()

        # Compute costs (convert tensor params to Python floats)
        C_I = self.C_I.item()
        C_V = self.C_V.item()
        C_O = self.C_O.item()
        infection_cost = C_I * num_infected
        vaccination_cost = C_V * num_vaccinated
        individual_cost = infection_cost + vaccination_cost
        facility_cost = C_O * num_open_val * num_dt
        cost = individual_cost + facility_cost

        # Get current compartment counts
        individual_x = self.data["individual"].x
        current_S = (individual_x[:, 0] == 1).sum().item()
        current_V = (individual_x[:, 2] == 1).sum().item()

        details = {
            'total_infections': num_infected,
            'total_vaccinations': num_vaccinated,
            'infection_cost': infection_cost,
            'vaccination_cost': vaccination_cost,
            'individual_cost': individual_cost,
            'operation_cost': facility_cost,
            'facility_cost': facility_cost,  # Keep for backward compatibility
            'current_infected': current_infected,
            'current_susceptible': current_S,
            'current_vaccinated': current_V,
        }

        return cost, None, details

    def mean_field(self, num_dt, visualize=True):

        def edge_index_to_adj_matrix(edge_index, num_nodes):
            adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            src, dst = edge_index
            for i, j in zip(src.tolist(), dst.tolist()):
                adj[i, j] = 1
            return adj

        def get_X0():
            x = self.data['individual'].x.cpu()
            S_mask = x[:, 0] == 1
            I_mask = x[:, 1] == 1
            V_mask = x[:, 2] == 1
            X0 = np.zeros(3 * x.size(0))
            X0[S_mask.nonzero(as_tuple=True)[0]] = 1  # S compartment
            X0[I_mask.nonzero(as_tuple=True)[0] + x.size(0)] = 1  # I compartment
            X0[V_mask.nonzero(as_tuple=True)[0] + 2 * x.size(0)] = 1  # V compartment
            return X0

        def deriv(X, t, adj_matrix, nu):
            N = self.N

            # Split compartments from X: [S_0..S_{N-1}, I_0..I_{N-1}, V_0..V_{N-1}]
            S = X[0:N]
            I = X[N:2 * N]
            V = X[2 * N:3 * N]


            beta_1 = float(self.beta_1)
            beta_2 = float(self.beta_2)
            delta = float(self.delta)
            omega = float(self.omega)

            # --- Compute infection pressure: sum_j I_j for each node ---
            sum_I_neighbors = adj_matrix @ I  # [N], using matrix-vector multiplication

            # --- Vectorized derivatives ---
            dS = -nu * S - beta_1 * S * sum_I_neighbors + omega * V
            dI = beta_1 * S * sum_I_neighbors + beta_2 * V * sum_I_neighbors - delta * I
            dV = nu * S + delta * I - omega * V - beta_2 * V * sum_I_neighbors

            return np.concatenate([dS, dI, dV])

        adj_matrix = edge_index_to_adj_matrix(
            self.data['individual', 'interacts', 'individual'].edge_index,
            self.data['individual'].num_nodes
        )
        X0 = get_X0()

        t = np.linspace(0, num_dt * self.dt, 1000)
        # Precompute nu outside odeint since it's time-invariant over the integration window
        nu_tensor = self.compute_vaccination_rates()
        nu = nu_tensor.detach().cpu().numpy()
        y = odeint(deriv, y0=X0, t=t, args=(adj_matrix, nu))

        S, I, V = y[:, 0: self.N], y[:, self.N: 2 * self.N], y[:, 2 * self.N:]

        if visualize:
            fig, ax = plt.subplots()
            ax.plot(t, S.mean(axis=1), linewidth=2, linestyle="--", color=self.compartments_colors["S"], label="Susceptible")
            ax.plot(t, I.mean(axis=1), linewidth=2, linestyle=":", color=self.compartments_colors["I"], label="Infected")
            ax.plot(t, V.mean(axis=1), linewidth=2, color="green", label="Vaccinated")
            plt.grid(linestyle="--", linewidth=1, alpha=0.5)
            plt.legend()
            plt.show()

        return S, I, V, t

    def save(self, filepath, metadata=None):
        """
        Save the EpidemicEnvironment to a file.

        Args:
            filepath: Path to save the environment (e.g., 'problem_001.pt')
            metadata: Optional dict with additional metadata (description, tags, etc.)

        Returns:
            filepath: The path where the environment was saved
        """
        # Move all tensors to CPU for portability
        data_cpu = self.data.clone().cpu()

        save_dict = {
            # Version for backward compatibility
            "version": ENVIRONMENT_SAVE_VERSION,

            # Timestamp
            "saved_at": datetime.datetime.now().isoformat(),

            # Core graph data (HeteroData object)
            "data": data_cpu,

            # Environment parameters (as plain Python dict)
            "env_params": copy.deepcopy(self.env_params),
            "env_params_range": copy.deepcopy(self.env_params_range),

            # Compartment definitions
            "compartments": {k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in self.compartments.items()},
            "compartments_colors": copy.deepcopy(self.compartments_colors),
            "compartments_abbr": copy.deepcopy(self.compartments_abbr),
            "init_compartments": copy.deepcopy(self.init_compartments),

            # Configuration flags
            "randomize": self.randomize,

            # Cached computations (as CPU tensor)
            "min_dists_to_open_facilities": self.min_dists_to_open_facilities.cpu(),

            # User-provided metadata
            "metadata": metadata or {},
        }

        # Ensure directory exists
        save_dir = os.path.dirname(filepath)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, filepath)
        return filepath

    @classmethod
    def load(cls, filepath, device=None):
        """
        Load an EpidemicEnvironment from a saved file.

        Args:
            filepath: Path to the saved environment file
            device: Target device (default: use global device from params.py)

        Returns:
            EpidemicEnvironment: Restored environment instance
        """
        if device is None:
            from params import device as default_device
            device = default_device

        save_dict = torch.load(filepath, map_location='cpu', weights_only=False)

        # Version check for backward compatibility
        saved_version = save_dict.get("version", "unknown")
        if saved_version != ENVIRONMENT_SAVE_VERSION:
            print(f"Warning: Loading environment saved with version {saved_version}, "
                  f"current version is {ENVIRONMENT_SAVE_VERSION}")

        # Reconstruct compartments with numpy arrays
        compartments_loaded = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in save_dict["compartments"].items()
        }

        # Create instance without calling __init__
        instance = cls.__new__(cls)

        # Initialize MessagePassing parent
        MessagePassing.__init__(instance, aggr='add')

        # Restore configuration
        instance.randomize = save_dict["randomize"]
        instance.env_params = save_dict["env_params"]
        instance.env_params_range = save_dict["env_params_range"]
        instance.compartments = compartments_loaded
        instance.compartments_colors = save_dict["compartments_colors"]
        instance.compartments_abbr = save_dict["compartments_abbr"]
        instance.init_compartments = save_dict["init_compartments"]

        # Restore scalar parameters as tensors on target device
        instance.N = instance.env_params["N"]
        instance.M = instance.env_params["M"]
        instance.beta_1 = torch.tensor(instance.env_params["beta_1"], device=device, dtype=torch.float32)
        instance.beta_2 = torch.tensor(instance.env_params["beta_2"], device=device, dtype=torch.float32)
        instance.delta = torch.tensor(instance.env_params["delta"], device=device, dtype=torch.float32)
        instance.omega = torch.tensor(instance.env_params["omega"], device=device, dtype=torch.float32)
        instance.v_min = torch.tensor(instance.env_params["v_min"], device=device, dtype=torch.float32)
        instance.v_max = torch.tensor(instance.env_params["v_max"], device=device, dtype=torch.float32)
        instance.alpha = torch.tensor(instance.env_params["alpha"], device=device, dtype=torch.float32)
        instance.f_plus = torch.tensor(instance.env_params["f_plus"], device=device, dtype=torch.float32)
        instance.f_minus = torch.tensor(instance.env_params["f_minus"], device=device, dtype=torch.float32)
        instance.C_O = torch.tensor(instance.env_params["C_O"], device=device, dtype=torch.float32)
        instance.C_I = torch.tensor(instance.env_params["C_I"], device=device, dtype=torch.float32)
        instance.C_V = torch.tensor(instance.env_params["C_V"], device=device, dtype=torch.float32)
        instance.network_type = instance.env_params["network_type"]
        instance.avg_deg = instance.env_params["avg_deg"]
        instance.pct_open_fac = instance.env_params["pct_open_fac"]
        instance.dt = instance.env_params["dt"]

        # Restore graph data and move to device
        instance.data = save_dict["data"].to(device)

        # Restore cached computations
        instance.min_dists_to_open_facilities = save_dict["min_dists_to_open_facilities"].to(device)

        # Store metadata for reference
        instance._loaded_metadata = save_dict.get("metadata", {})
        instance._loaded_from = filepath
        instance._saved_at = save_dict.get("saved_at")

        return instance

    def get_state_dict(self):
        """
        Get a dictionary representation of the current state (useful for checkpointing).

        Returns:
            dict: State dictionary containing all environment state
        """
        return {
            "individual_x": self.data['individual'].x.clone(),
            "facility_x": self.data['facility'].x.clone(),
            "min_dists": self.min_dists_to_open_facilities.clone(),
        }

    def set_state_dict(self, state_dict):
        """
        Restore the environment to a previous state.

        Args:
            state_dict: Dictionary from get_state_dict()
        """
        self.data['individual'].x = state_dict["individual_x"].clone()
        self.data['facility'].x = state_dict["facility_x"].clone()
        self.min_dists_to_open_facilities = state_dict["min_dists"].clone()

    def reset(self):
        """
        Reset the environment to a fresh initial state (regenerate graph).
        Useful for RL episode resets.

        Returns:
            self: The reset environment
        """
        self.data = self.generate_graph()
        self.min_dists_to_open_facilities = self._compute_min_dists_to_open_facilities()
        return self


class EpidemicProblemDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading saved EpidemicEnvironment problems.
    Compatible with torch.utils.data.DataLoader for batched RL training.

    Usage:
        # Load all problems from a directory
        dataset = EpidemicProblemDataset('problems/')
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for batch in loader:
            # batch is a list of EpidemicEnvironment instances
            for env in batch:
                # Train RL agent on env
                pass
    """

    def __init__(self, problem_dir, pattern="*.pt", device=None, load_on_access=True):
        """
        Initialize the dataset.

        Args:
            problem_dir: Directory containing saved problem files
            pattern: Glob pattern for problem files (default: "*.pt")
            device: Device to load environments to (default: use global device)
            load_on_access: If True, load environments lazily on __getitem__.
                           If False, preload all environments into memory.
        """

        self.problem_dir = problem_dir
        self.device = device
        self.load_on_access = load_on_access

        # Find all matching files
        search_path = os.path.join(problem_dir, pattern)
        self.problem_files = sorted(glob_module.glob(search_path))

        if len(self.problem_files) == 0:
            raise ValueError(f"No files matching '{pattern}' found in '{problem_dir}'")

        # Optionally preload all environments
        self._cache = {}
        if not load_on_access:
            print(f"Preloading {len(self.problem_files)} environments...")
            for i, filepath in enumerate(tqdm(self.problem_files)):
                self._cache[i] = EpidemicEnvironment.load(filepath, device=self.device)

    def __len__(self):
        return len(self.problem_files)

    def __getitem__(self, idx):
        """
        Get an environment by index.

        Args:
            idx: Index of the problem

        Returns:
            EpidemicEnvironment: The loaded environment
        """
        if idx in self._cache:
            return self._cache[idx]

        env = EpidemicEnvironment.load(self.problem_files[idx], device=self.device)

        if not self.load_on_access:
            self._cache[idx] = env

        return env

    def get_filepath(self, idx):
        """Get the filepath for a problem by index."""
        return self.problem_files[idx]

    def get_metadata(self, idx):
        """Get metadata for a problem without fully loading the environment."""
        save_dict = torch.load(self.problem_files[idx], map_location='cpu', weights_only=False)
        return {
            "filepath": self.problem_files[idx],
            "version": save_dict.get("version"),
            "saved_at": save_dict.get("saved_at"),
            "metadata": save_dict.get("metadata", {}),
            "env_params": save_dict.get("env_params", {}),
        }


def generate_problem_set(output_dir, num_problems, randomize=True, seed=None,
                         env_params=None, env_params_range=None,
                         metadata_fn=None, verbose=True):
    """
    Generate and save a set of test problems.

    Args:
        output_dir: Directory to save problems
        num_problems: Number of problems to generate
        randomize: Whether to randomize parameters (default: True)
        seed: Random seed for reproducibility (optional)
        env_params: Base environment parameters (optional, uses default if None)
        env_params_range: Parameter ranges for randomization (optional)
        metadata_fn: Callable(index, env) -> dict that generates metadata for each problem
        verbose: Whether to print progress

    Returns:
        list: List of saved filepaths
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    filepaths = []
    iterator = range(num_problems)
    if verbose:
        iterator = tqdm(iterator, desc="Generating problems")

    for i in iterator:
        # Build kwargs for EpidemicEnvironment
        kwargs = {"randomize": randomize}
        if env_params is not None:
            kwargs["env_params"] = env_params
        if env_params_range is not None:
            kwargs["env_params_range"] = env_params_range

        env = EpidemicEnvironment(**kwargs)

        # Generate metadata
        metadata = {
            "problem_index": i,
            "seed": seed,
            "total_problems": num_problems,
        }
        if metadata_fn is not None:
            metadata.update(metadata_fn(i, env))

        # Save with zero-padded index
        filename = f"problem_{i:06d}.pt"
        filepath = os.path.join(output_dir, filename)
        env.save(filepath, metadata=metadata)
        filepaths.append(filepath)

    if verbose:
        print(f"Generated {num_problems} problems in '{output_dir}'")

    return filepaths


def load_problem_info(filepath):
    """
    Load basic info about a saved problem without fully loading the environment.

    Args:
        filepath: Path to the saved problem file

    Returns:
        dict: Problem information including parameters and metadata
    """
    save_dict = torch.load(filepath, map_location='cpu', weights_only=False)
    return {
        "filepath": filepath,
        "version": save_dict.get("version"),
        "saved_at": save_dict.get("saved_at"),
        "metadata": save_dict.get("metadata", {}),
        "env_params": save_dict.get("env_params", {}),
        "N": save_dict["env_params"].get("N"),
        "M": save_dict["env_params"].get("M"),
        "network_type": save_dict["env_params"].get("network_type"),
    }


def validate_saved_environment(filepath, verbose=True):
    """
    Validate that a saved environment can be loaded correctly.

    Args:
        filepath: Path to the saved problem file
        verbose: Whether to print validation results

    Returns:
        tuple: (is_valid, error_message or None)
    """
    try:
        # Try to load
        env = EpidemicEnvironment.load(filepath)

        # Basic checks
        assert env.data is not None, "Graph data is None"
        assert env.data['individual'].x is not None, "Individual features are None"
        assert env.data['facility'].x is not None, "Facility features are None"
        assert env.N == env.data['individual'].x.size(0), "N mismatch"
        assert env.M == env.data['facility'].x.size(0), "M mismatch"

        # Try a forward pass
        _ = env.forward(
            env.data["individual"].x,
            env.data['individual', 'interacts', 'individual'].edge_index,
            env.data["facility"].x,
            env.data['individual', 'visits', 'facility'].edge_index
        )

        if verbose:
            print(f"Validation passed: {filepath}")
        return True, None

    except Exception as e:
        error_msg = f"Validation failed: {str(e)}"
        if verbose:
            print(f"{error_msg} for {filepath}")
        return False, error_msg

