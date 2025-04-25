data {
    // Test
    int<lower=1> M; // Number of participants
    int<lower=1> N; // Number of test observations per participant
    int<lower=1> N_test; // Number of test observations per participant
    int<lower=1> K; // Number of trials per block

    array[M, N, K] int certainties; // certainty conditions
    array[M, N_test] int certainties_test; // certainty conditions
    int<lower=1> C; // Number of certainty conditions
    
    array[M, N, K] real trusts; // trust conditions
    array[M, N_test] int trusts_test; // trust conditions
    int<lower=1> T; // Number of trusts

    int<lower=1> A; // Number of actions
    array[M, N, K] int y; // Observations
    array[M, N_test] int y_test; // Test observations
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
    real trust_w; // Trust weight
    real<lower=0> sigma_trust_w; // Trust weight
    //// Interactions
    array[C] real certainty_trust_w; // Trust weights
    real<lower=0> sigma_certainty_trust_w; // Trust weight

    //// Action (boss) weight
    real boss_w; // Action weight
    real<lower=0> sigma_boss_w; // Action weight

    // Parameters individual-level
    //// Utility weights
    array[M, C] real certainty_w_M; // Mean of the distribution of certainty weights
    array[M] real trust_w_M; // Mean of the distribution of trust weights
    array[M, C] real certainty_trust_w_M; // Mean of the distribution of trust weights
    array[M] real boss_w_M; // Mean of the distribution of boss weights
}

model {
    // Priors group-level
    //// Utility weights
    certainty_w ~ normal(0, 1); // Mean of the distribution of certainty_w
    sigma_certainty_w ~ exponential(1);
    trust_w ~ normal(0, 1); // Mean of the distribution of beta (prosocial)
    sigma_trust_w ~ exponential(1);
    // Certainty-trust interaction
    certainty_trust_w ~ normal(0, 1); // Mean of the distribution of certainty-trust
    sigma_certainty_trust_w ~ exponential(1);

    boss_w ~ normal(0, 1); // Mean of the distribution of action (presentational)
    sigma_boss_w ~ exponential(1);

    // Priors individual-level
    //// Utility weights
    for (c in 1:C) {
        certainty_w_M[, c] ~ normal(0, 1); // Mean of the distribution of certainty
        certainty_trust_w_M[, c] ~ normal(0, 1); // Mean of the distribution of certainty-trust
    }
    //certainty_w_M ~ normal(0, 1); // Mean of the distribution of certainty
    trust_w_M ~ normal(0, 1); // Mean of the distribution of trust
    //certainty_trust_w_M ~ normal(0, 1); // Mean of the distribution of certainty-trust
    boss_w_M ~ normal(0, 1); // Mean of the distribution of boss


    // Compute model likelihood
    // Loop over partipants
    for (m in 1:M) {
        // Compute individual-level parameters
        // Boss face weight
        real boss_w_m = boss_w + sigma_boss_w * boss_w_M[m];

        // Certainty
        array[C] real certainty_w_m;
        array[C] real certainty_trust_w_m;
        for (c in 1:C) {
            certainty_w_m[c] = certainty_w[c] + sigma_certainty_w * certainty_w_M[m, c];
            certainty_trust_w_m[c] = certainty_trust_w[c] + sigma_certainty_trust_w * certainty_trust_w_M[m, c];
        }
        // Trust
        real trust_w_m = trust_w + sigma_trust_w * trust_w_M[m];
    
        
        // Loop over participant's trials
        for (n in 1:N) {
            // Loop over observations
            for (k in 1:K) {
                // Extract trial data
                int action = y[m, n, k];
                real trust = trusts[m, n, k];
                int certainty = certainties[m, n, k];

                // Compute function value to be evaluated for action selection
                vector[A] utility_A;
                for (a in 1:A) {
                    // Compute function value to be evaluated for action selection
                    utility_A[a] = rewards_G[a] * boss_w_m + certainty_w_m[certainty] + trust_w_m * trust + certainty_trust_w_m[certainty] * trust;
                }

                // Increment the log likelihood
                y[m, n, k] ~ categorical_logit(utility_A);
            }
        }
    }
}

generated quantities {

    // Participant parameters
    array[M] real boss_w_M_sim;
    array[M, C] real certainty_w_M_sim;
    array[M] real trust_w_M_sim;
    array[M, C] real certainty_trust_w_M_sim;

    array[M, N_test] real y_sim;

    // Loop over partipants  
    for (m in 1:M) {
        real boss_w_m = boss_w + sigma_boss_w * boss_w_M[m];
        boss_w_M_sim[m] = boss_w_m;
        
        // Certainty
        array[C] real certainty_w_m;
        array[C] real certainty_trust_w_m;
        for (c in 1:C) {
            certainty_w_m[c] = certainty_w[c] + sigma_certainty_w * certainty_w_M[m, c];
            certainty_w_M_sim[m, c] = certainty_w_m[c];

            certainty_trust_w_m[c] = certainty_trust_w[c] + sigma_certainty_trust_w * certainty_trust_w_M[m, c];
            certainty_trust_w_M_sim[m, c] = certainty_trust_w_m[c];
        }
        // Trust
        
        real trust_w_m = trust_w + sigma_trust_w * trust_w_M[m];
        trust_w_M_sim[m] = trust_w_m;


        // Loop over participant's trials
        for (n in 1:N_test) {
            // Extract trial data
            int trust = trusts_test[m, n];
            int certainty = certainties_test[m, n];

            vector[A] utility_A;
            for (a in 1:A) {
                // Compute function value to be evaluated for action selection
                utility_A[a] = rewards_G[a] * boss_w_m + certainty_w_m[certainty] + trust_w_m * trust + certainty_trust_w_m[certainty] * trust;
            }
            
            // Sample from the utility distribution
            y_sim[m, n] = categorical_logit_rng(utility_A);
        }
    }
}