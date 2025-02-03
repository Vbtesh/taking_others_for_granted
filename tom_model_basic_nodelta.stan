data {
    int<lower=1> N; // Number of observations per participant
    int<lower=1> M; // Number of participants

    array[M, N] int certainties; // certainty conditions
    int<lower=1> C; // Number of certainty conditions
    
    array[M, N] int trusts; // trust conditions
    int<lower=1> T; // Number of trusts

    int<lower=1> A; // Number of actions
    array[M, N] int y; // Observations
}

transformed data {
    int<lower=1> N_total = N * M; // Total number of observations
    int S = 100; // Number of simulations

    // Actions rewards
    vector[4] rewards_G = [0, 1, 2, 3]';
    vector[4] rewards_R = [4, 4, 0, 0]';
    
    // Alpha value (constant)
    real alpha_constant = 0.75;
    real beta_constant = 0;
    real delta_constant = 0;

    // beta values
    vector[3] beta_values = [-1, 0, 1]';

    // Base prior distribution for trust conditions
    simplex[3] prior_betaB_low = [0.7, 0.2, 0.1]';
    simplex[3] prior_betaB_medium = [0.15, 0.7, 0.15]';
    simplex[3] prior_betaB_high = [0.1, 0.2, 0.7]';

    array[3] simplex[3] prior_betaB;
    prior_betaB[1] = prior_betaB_low;
    prior_betaB[2] = prior_betaB_medium;
    prior_betaB[3] = prior_betaB_high;

    // Giver's preference over the Receiver's beliefs of them
    simplex[3] rho_betaB = [0.05, 0.15, 0.8]';
}

parameters {
    // Parameters group-level
    //// Utility weights
    real<lower=0> alpha_g; // Mean of the distribution of alpha
    real<lower=0> sigma_alpha; 
    real beta_g; // Mean of the distribution of beta (prosocial)
    real<lower=0> sigma_beta;
    //real<lower=0> delta_g; // Mean of the distribution of delta (presentational)
    //real<lower=0> sigma_delta;

    //// Prior inverse temperatures for certainty conditions
    array[C] real prior_inv_temps_C; // Inverse temperature for prior [1: uncertain, 2: certain, 3: immutable]

    // Parameters individual-level
    //// Utility weights
    array[M] real alpha_M; // Mean of the distribution of alpha
    array[M] real beta_M; // Mean of the distribution of beta (prosocial)
    //array[M] real delta_M; // Mean of the distribution of delta (presentational)
    //// Prior entropy
    array[M] real prior_inv_temp_M; // Inverse temperature for prior

}

model {
    // Priors group-level
    //// Utility weights
    alpha_g ~ lognormal(0, 1); // Mean of the distribution of alpha
    sigma_alpha ~ exponential(1);
    beta_g ~ normal(0, 1); // Mean of the distribution of beta (prosocial)
    sigma_beta ~ exponential(1);
    //delta_g ~ lognormal(0, 1); // Mean of the distribution of delta (presentational)
    //sigma_delta ~ exponential(1);
    //// Prior entropies for certainty condition
    prior_inv_temps_C ~ normal(0, 1); // Inverse temperature for prior

    // Priors individual-level
    //// Utility weights
    alpha_M ~ normal(0, 1); // Mean of the distribution of alpha
    beta_M ~ normal(0, 1); // Mean of the distribution of beta (prosocial)
    //delta_M ~ normal(0, 1); // Mean of the distribution of delta (presentational)
    //// Prior entropy
    prior_inv_temp_M ~ normal(0, 1); // Inverse temperature for prior

    // Compute model likelihood
    // Loop over partipants
    for (m in 1:M) {
        // Compute individual-level parameters
        //// Utility weights
        real alpha_m = alpha_g + alpha_M[m] * sigma_alpha;
        real beta_m = beta_g + beta_M[m] * sigma_beta;
        real delta_m = delta_constant; //delta_g + delta_M[m] * sigma_delta;

        //// Prior inverse temperature for certainty conditions
        array[C] real prior_inv_temps_m;
        for (c in 1:C) {
            prior_inv_temps_m[c] = exp(prior_inv_temps_C[c] + prior_inv_temp_M[m]);
        }

        // Loop over participant's trials
        for (n in 1:N) {
            // Extract trial data
            int trust = trusts[m, n];
            int certainty = certainties[m, n];
            real prior_inv_temp_mc = prior_inv_temps_m[certainty];
            vector[3] prior_betaB_c;

            // Define prior distriubtion
            if (trust == 4) {
                // Trust is none, so no certainty
                prior_betaB_c = prior_betaB[1];
            } else {
                // Recover trust specific prior
                vector[3] prior_betaB_t = prior_betaB[trust];
                // Adjust with certainty
                for (k in 1:3) {
                    prior_betaB_c[k] = prior_inv_temp_mc * prior_betaB_t[k];
                }
                // Apply softmax
                prior_betaB_c = softmax(prior_betaB_c);
            }
            // UTILITY
            //// For each action, compute conditional posterior and KL distribution with prior
            //// Need prior distribution and utility weights
            
            array[A] vector[3] posterior_betaB_A;
            vector[A] kl_betaB_A;
            vector[A] utility_A;
            for (a in 1:A) {
                // Compute conditional posterior
                if (trust == 4) {
                    // Trust is none, so no change in beliefs about betaB
                    posterior_betaB_A[a] = prior_betaB_c;
                } else {
                    // Compute likelihood of A given prior parameters
                    vector[3] log_posterior_a;
                    for (u in 1:3) {
                        // Assume alpha=0.5
                        log_posterior_a[u] = alpha_constant * rewards_G[a] + beta_values[u] * rewards_R[a] + log(prior_betaB_c[u]);
                    }
                    // Compute posterior
                    posterior_betaB_A[a] = softmax(log_posterior_a);
                } 
            }

            for (a in 1:A) {
                // Compute KL divergence
                vector[3] kl_vals_a;
                for (u in 1:3) {
                    real posterior_betaB_au;
                    // Average over 2 middle actions to represent road 2
                    if (a == 2 || a == 3) {
                        posterior_betaB_au = (posterior_betaB_A[2][u] + posterior_betaB_A[3][u]) / 2;
                    } else {
                        posterior_betaB_au = posterior_betaB_A[a][u];
                    }
                    // Compute KL divergence term
                    kl_vals_a[u] = posterior_betaB_au * log((posterior_betaB_au + 1e-10) / (rho_betaB[u] + 1e-10));
                }
                kl_betaB_A[a] = sum(kl_vals_a);
        
                // Compute rest of utility using participant's utility weights
                utility_A[a] = alpha_m * rewards_G[a] + beta_m * rewards_R[a] - delta_m * kl_betaB_A[a];
            }

            // Increment the log likelihood
            y[m, n] ~ categorical_logit(utility_A);    
        }
    }
}

generated quantities {

    // Participant parameters
    array[M] real alpha_M_sim;
    array[M] real beta_M_sim;
    //array[M] real delta_M_sim;
    array[M, C] real prior_inv_temp_M_sim;
    array[M, N] real y_sim;
    array[M, N] real y_nobeta_sim;
        
    for (m in 1:M) {
        real alpha_m = alpha_g + alpha_M[m] * sigma_alpha;
        real beta_m = beta_g + beta_M[m] * sigma_beta;
        real delta_m = delta_constant; //delta_g + delta_M[m] * sigma_delta;
        //// Prior inverse temperature for certainty conditions
        array[C] real prior_inv_temps_m;
        for (c in 1:C) {
            prior_inv_temps_m[c] = exp(prior_inv_temps_C[c] + prior_inv_temp_M[m]);
            // Set prior inverse temperature for sim
            prior_inv_temp_M_sim[m, c] = prior_inv_temps_m[c];
        }
        // Alpha beta and delta sims for participants
        alpha_M_sim[m] = alpha_m;
        beta_M_sim[m] = beta_m;
        //delta_M_sim[m] = delta_m;

        // Loop over participant's trials
        for (n in 1:N) {
            // Extract trial data
            int trust = trusts[m, n];
            int certainty = certainties[m, n];
            real prior_inv_temp_mc = prior_inv_temps_m[certainty];
            vector[3] prior_betaB_c;
            // Define prior distriubtion
            if (trust == 4) {
                // Trust is none, so no certainty
                prior_betaB_c = prior_betaB[1];
            } else {
                // Recover trust specific prior
                vector[3] prior_betaB_t = prior_betaB[trust];
                // Adjust with certainty
                for (k in 1:3) {
                    prior_betaB_c[k] = prior_inv_temp_mc * prior_betaB_t[k];
                }
                // Apply softmax
                prior_betaB_c = softmax(prior_betaB_c);
            }
            // UTILITY
            //// For each action, compute conditional posterior and KL distribution with prior
            //// Need prior distribution and utility weights
            array[A] vector[3] posterior_betaB_A;
            vector[A] kl_betaB_A;
            vector[A] utility_A;
            vector[A] utility_nobeta_A;
            for (a in 1:A) {
                if (trust != 4) {
                    // Compute likelihood of A given prior parameters
                    vector[3] log_posterior_a;
                    for (u in 1:3) {
                        // Assume alpha=0.5
                        log_posterior_a[u] = alpha_constant * rewards_G[a] + beta_values[u] * rewards_R[a] + log(prior_betaB_c[u]);
                    }
                    // Compute posterior
                    posterior_betaB_A[a] = softmax(log_posterior_a);
                } else {
                    // Trust is none, so no change in beliefs about betaB
                    posterior_betaB_A[a] = prior_betaB_c;
                }
            }

            for (a in 1:A) {
                // Compute KL divergence
                vector[3] kl_vals_a;
                for (u in 1:3) {
                    // Average over 2 middle actions to represent road 2
                    real posterior_betaB_au;
                    if (a == 2 || a == 3) {
                        posterior_betaB_au = (posterior_betaB_A[2][u] + posterior_betaB_A[3][u]) / 2;
                    } else {
                        posterior_betaB_au = posterior_betaB_A[a][u];
                    }
                    // Compute KL divergence term
                    kl_vals_a[u] = posterior_betaB_au * log((posterior_betaB_au + 1e-10) / (rho_betaB[u] + 1e-10));
                }
                kl_betaB_A[a] = sum(kl_vals_a);
        
                // Compute rest of utility using participant's utility weights
                utility_A[a] = alpha_m * rewards_G[a] + beta_m * rewards_R[a] - delta_m * kl_betaB_A[a];
                // Compute utility without beta
                utility_nobeta_A[a] = alpha_m * rewards_G[a] - delta_m * kl_betaB_A[a];
            }

            // Sample from the utility distribution
            y_sim[m, n] = categorical_logit_rng(utility_A);
            // Sample from the utility distribution without beta
            y_nobeta_sim[m, n] = categorical_logit_rng(utility_nobeta_A);
        }
    }
}