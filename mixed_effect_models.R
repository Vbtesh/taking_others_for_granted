library(afex)
library(emmeans)
library(parallel)

(nc = detectCores())
cl <- makeCluster(rep("localhost", nc))

df_all = read.csv('./data/pilot/df_long_test_cluster.csv')
df_all$cluster = as.factor(df_all$cluster)
df_all$participant_code = as.factor(df_all$participant_code)
df_all$certainty = factor(df_all$certainty, levels=c('uncertain', 'certain', 'immutable'))
df_all$intention = as.factor(df_all$intention)
df_all$trust_cond = factor(df_all$trust_cond, levels=c('high', 'medium', 'low'))
df_all$absent = as.logical(df_all$absent)


# Differences between absent and non absent trials
modelAbsent = afex::mixed(path_length ~ absent + (absent|participant_code), data=df_all, cl=cl)

modelAbsent
afex_plot(modelAbsent, 'absent')
em = emmeans::emmeans(modelAbsent, ~absent)
em
pairs(em)

# Differences between absent and non absent trials in probability of delivering the letter
modelAbsent = afex::mixed(delivered ~ absent + (1|participant_code), data=df_all, cl=cl, family=binomial(link="logit"),
                          method = "LRT")

modelAbsent
afex_plot(modelAbsent, 'absent')
em = emmeans::emmeans(modelAbsent, ~absent)
em
pairs(em)



# Rest of behavioural analysis for non absent trials 
df = df_all[df_all$absent == FALSE,]

# Path as continuous variable
## NO CLUSTER
modelLength = afex::mixed(path_length ~ certainty*trust_cond + (trust_cond|participant_code), data=df, cl=cl)

modelLength
afex_plot(modelLength, 'trust_cond', 'certainty')
em = emmeans::emmeans(modelLength, ~certainty)
em
pairs(em)
em = emmeans::emmeans(modelLength, ~trust_cond)
em
pairs(em)

## CLUSTER
modelLengthC = afex::mixed(path_length ~ certainty*trust_cond*cluster + (trust_cond|participant_code), data=df, cl=cl)

modelLengthC
afex_plot(modelLengthC, 'trust_cond', 'certainty', 'cluster')
em = emmeans::emmeans(modelLengthC, ~certainty*cluster*trust_cond, , where=c(trust_cond='low'))
em
pairs(em)
em = emmeans::emmeans(modelLengthC, ~cluster*trust_cond)
em
pairs(em)

# Letter delivered
## No cluster
modelDelivered = afex::mixed(delivered ~ certainty*trust_cond + (1|participant_code), data=df, family=binomial(link="logit"),
                     method = "LRT", cl=cl)

modelDelivered
afex_plot(modelDelivered, 'trust_cond', 'certainty')
em = emmeans::emmeans(modelDelivered, ~certainty, reverse = TRUE)
em
pairs(em)
em = emmeans::emmeans(modelDelivered, ~trust_cond)
em
pairs(em)

## CLUSTER
modelDeliveredC = afex::mixed(delivered ~ certainty*trust_cond*cluster + (1|participant_code), data=df, family=binomial(link="logit"),
                             method = "LRT", cl=cl)

modelDeliveredC
afex_plot(modelDeliveredC, 'trust_cond', 'certainty', 'cluster')
em = emmeans::emmeans(modelDeliveredC, ~certainty, reverse = TRUE)
em
pairs(em)
em = emmeans::emmeans(modelDeliveredC, ~trust_cond)
em
pairs(em)

# Hidden path
modelHidden = afex::mixed(path_hidden ~ certainty*trust_cond + (1|participant_code), data=df, family=binomial(link="logit"),
                             method = "LRT", cl=cl)

modelHidden
afex_plot(modelHidden, 'trust_cond', 'certainty')

### FOR EACH ACTION
# PATH A
modelA = afex::mixed(A ~ certainty*trust_cond + (1|participant_code), data=df, family=binomial(link="logit"),
                     method = "LRT", cl=cl)

modelA
summary(modelA)
afex_plot(modelA, 'trust_cond', 'certainty')
# PATH B
modelB = afex::mixed(B ~ certainty*trust_cond + (trust_cond||participant_code), data=df, family=binomial(link="logit"),
                     method = "LRT", expand_re = TRUE, cl=cl)
modelB
afex_plot(modelB, 'trust_cond', 'certainty')
# PATH C
modelC = afex::mixed(C ~ certainty*trust_cond + (trust_cond||participant_code), data=df, family=binomial(link="logit"),
                     method = "LRT", expand_re = TRUE, cl=cl)
modelC
afex_plot(modelC, 'trust_cond', 'certainty')
# PATH D
modelD = afex::mixed(D ~ certainty*trust_cond + (certainty + trust_cond||participant_code), data=df, family=binomial(link="logit"),
                     method = "LRT", expand_re = TRUE, cl=cl)
modelD
afex_plot(modelD, 'trust_cond', 'certainty')
