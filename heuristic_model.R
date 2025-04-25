library(afex)
library(emmeans)
library(parallel)

(nc = detectCores())
cl <- makeCluster(rep("localhost", nc))

df_all = read.csv('./data/pilot/design_matrix_test.csv')
df_all$participant_code = as.factor(df_all$participant_code)
df_all$certainty = factor(df_all$certainty, levels=c('uncertain', 'certain', 'immutable'))
df_all$trust = factor(df_all$trust, levels=c('high', 'medium', 'low'))
df_all$path_length = as.numeric(df_all$path_length)

# Remove df_all trust == None
df = df_all[df_all$trust_lvl != 4,]
# Path as continuous variable
modelLength = afex::mixed(path_length ~ certainty*trust + (trust|participant_code), data=df, cl=cl)

modelLength
afex_plot(modelLength, 'trust_cond', 'certainty')
em = emmeans::emmeans(modelLength, ~certainty)
em
pairs(em)
em = emmeans::emmeans(modelLength, ~trust_cond)
em
pairs(em)

# Predict the dataset from the model

modelPred = predict(modelLength$full_model, newdata=df)
dfPred = df
dfPred$preds = modelPred

write.csv(dfPred, 'data/simulations/heuristic_mixed_model_preds.csv')
