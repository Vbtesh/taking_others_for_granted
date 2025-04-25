data {
    // Test
    int<lower=1> N; // Number of test observations per participant
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
}

parameters {
    // Parameters group-level
    //// Certainty weights
    array[C] real certainty_w; //Certainty weights
    real<lower=0> sigma_certainty_w; // Certainty weight
    //// Face weight
    array[T] real trust_w; // Trust weight
    real<lower=0> sigma_trust_w; // Trust weight
    //// Interaction Trust x Certainty
    array[T, C] real interaction_w; // Interaction weight
    real<lower=0> sigma_interaction_w; // Interaction weight

    //// Action (boss) weight
    real boss_w; // Action weight
    real<lower=0> sigma_boss_w; // Action weight

    // Parameters individual-level
    //// Utility weights
    array[M] real certainty_w_M; // Mean of the distribution of certainty weights
    array[M] real trust_w_M; // Mean of the distribution of trust weights
    array[M] real interaction_w_M; // Mean of the distribution of interaction weights
    array[M] real boss_w_M; // Mean of the distribution of boss weights

}

model {
    // Priors group-level
    //// Utility weights
    certainty_w ~ normal(0, 1); // Mean of the distribution of certainty_w
    sigma_certainty_w ~ exponential(1);
    trust_w ~ normal(0, 1); // Mean of the distribution of beta (prosocial)
    sigma_trust_w ~ exponential(1);
    interaction_w ~ multi_normal(array[], 1); // Mean of the distribution of interaction
    sigma_interaction_w ~ exponential(1);

    boss_w ~ normal(0, 1); // Mean of the distribution of action (presentational)
    sigma_boss_w ~ exponential(1);

    // Priors individual-level
    //// Utility weights
    certainty_w_M ~ normal(0, 1); // Mean of the distribution of certainty
    trust_w_M ~ normal(0, 1); // Mean of the distribution of trust
    boss_w_M ~ normal(0, 1); // Mean of the distribution of boss


    // Compute model likelihood
    // Loop over partipants
    for (m in 1:M) {
        // Compute individual-level parameters
        // Boss face weight
        real boss_w_m = boss_w + sigma_boss_w * boss_w_M[m];

        // Certainty, trust and interaction
        array[C] real certainty_w_m;
        array[T] real trust_w_m;
        array[C, T] real interaction_w_m;
        for (c in 1:C) {
            certainty_w_m[c] = certainty_w[c] + sigma_certainty_w * certainty_w_M[m];
            for (t in 1:T) {
                trust_w_m[t] = trust_w[t] + sigma_trust_w * trust_w_M[m];
                interaction_w_m[c, t] = interaction_w[c, t] + sigma_interaction_w * interaction_w_M[m];
            }
        }
        
        // Loop over participant's trials
        for (n in 1:N) {
            // Extract trial data
            int trust = trusts[m, n];
            int certainty = certainties[m, n];

            vector[A] utility_A;
            for (a in 1:A) {
                // Compute function value to be evaluated for action selection
                utility_A[a] = rewards_G[a] * boss_w_m + certainty_w_m[certainty] + trust_w_m[trust] + interaction_w_m[certainty, trust];
            }

            // Increment the log likelihood
            y[m, n] ~ categorical_logit(utility_A);    
        }
    }
}

generated quantities {

    // Participant parameters
    array[M] real boss_w_M_sim;
    array[M, C] real certainty_w_M_sim;
    array[M, T] real trust_w_M_sim;
    array[M, C, T] real interaction_w_M_sim;

    array[M, N] real y_sim;

    // Loop over partipants  
    for (m in 1:M) {
        real boss_w_m = boss_w + sigma_boss_w * boss_w_M[m];
        boss_w_M_sim[m] = boss_w_m;
        
        // Certainty, trust and interaction
        array[C] real certainty_w_m;
        array[T] real trust_w_m;
        array[C, T] real interaction_w_m;
        for (c in 1:C) {
            certainty_w_m[c] = certainty_w[c] + sigma_certainty_w * certainty_w_M[m];
            certainty_w_M_sim[m, c] = certainty_w_m[c];
            for (t in 1:T) {
                trust_w_m[t] = trust_w[t] + sigma_trust_w * trust_w_M[m];
                interaction_w_m[c, t] = interaction_w[c, t] + sigma_interaction_w * interaction_w_M[m];
                trust_w_M_sim[m, t] = trust_w_m[t];
                interaction_w_M_sim[m, c, t] = interaction_w_m[c, t];
            }
        }

        // Loop over participant's trials
        for (n in 1:N) {
            // Extract trial data
            int trust = trusts[m, n];
            int certainty = certainties[m, n];

            vector[A] utility_A;
            for (a in 1:A) {
                // Compute function value to be evaluated for action selection
                utility_A[a] = rewards_G[a] * boss_w_m + certainty_w_m[certainty] + trust_w_m[trust] + interaction_w_m[certainty, trust];
            }
            
            // Sample from the utility distribution
            y_sim[m, n] = categorical_logit_rng(utility_A);
        }
    }
}