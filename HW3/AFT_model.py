import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lifelines import (
    WeibullAFTFitter, LogLogisticAFTFitter, LogNormalAFTFitter,
    GeneralizedGammaRegressionFitter, PiecewiseExponentialRegressionFitter,
)

df = pd.read_csv('telco.csv')
df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})
cat_cols = ['region', 'marital', 'ed', 'retire', 'gender', 'voice', 'internet', 'forward', 'custcat']

# Convert categorical cols to numeric codes
for col in cat_cols:
    df[col] = pd.Categorical(df[col]).codes

# Define features and target
X = df.drop(columns=['ID', 'tenure', 'churn'])
y = df['churn']
duration = df['tenure']

# Optionally use one-hot encoding for categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

df_model = X_encoded.copy()
df_model['churn'] = y
df_model['tenure'] = duration

duration_col = 'tenure'
event_col = 'churn'

# Weibull AFT model
weibull_aft = WeibullAFTFitter()
weibull_aft.fit(df_model, duration_col=duration_col, event_col=event_col)
print("Weibull AFT model summary:")
print(weibull_aft.summary)

# Log-Logistic AFT model
loglogistic_aft = LogLogisticAFTFitter()
loglogistic_aft.fit(df_model, duration_col=duration_col, event_col=event_col)
print("Log-Logistic AFT model summary:")
print(loglogistic_aft.summary)

# Log-Normal AFT model
lognormal_aft = LogNormalAFTFitter()
lognormal_aft.fit(df_model, duration_col=duration_col, event_col=event_col)
print("Log-Normal AFT model summary:")
print(lognormal_aft.summary)

# Exclude 'tenure' and 'churn' from features for VIF calculation
features = df_model.drop(columns=['tenure', 'churn'])

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = features.columns
vif_data["VIF"] = [variance_inflation_factor(features.values, i)
                   for i in range(features.shape[1])]

print(vif_data.sort_values(by='VIF', ascending=False))


# Drop highly correlated features
features_to_drop = ['age', 'custcat']
df_reduced = df_model.drop(columns=features_to_drop)

df_reduced['tenure_scaled'] = df_reduced['tenure'] / 100

# Generalized Gamma Regression with penalizer and alternate optimizer
gen_gamma_reg = GeneralizedGammaRegressionFitter(penalizer=0.01)
gen_gamma_reg._scipy_fit_method = "SLSQP"
gen_gamma_reg.fit(df_reduced, duration_col='tenure_scaled', event_col='churn')
print(gen_gamma_reg.summary)


# Piecewise Exponential Regression model
piecewise_exp_reg = PiecewiseExponentialRegressionFitter(breakpoints=[10, 20, 30])
piecewise_exp_reg.fit(df_model, duration_col=duration_col, event_col=event_col)
print("Piecewise Exponential Regression model summary:")
print(piecewise_exp_reg.summary)


# Plot survival functions for each model at average covariate values
plt.figure(figsize=(10, 6))

# Weibull survival function
weibull_sf = weibull_aft.predict_survival_function(df_model.drop(columns=['tenure', 'churn'])).mean(axis=1)
plt.step(weibull_sf.index, weibull_sf.values, label='Weibull AFT', color='blue')

# Log-Logistic survival function
loglogistic_sf = loglogistic_aft.predict_survival_function(df_model.drop(columns=['tenure', 'churn'])).mean(axis=1)
plt.step(loglogistic_sf.index, loglogistic_sf.values, label='Log-Logistic AFT', color='green')

# Log-Normal survival function
lognormal_sf = lognormal_aft.predict_survival_function(df_model.drop(columns=['tenure', 'churn'])).mean(axis=1)
plt.step(lognormal_sf.index, lognormal_sf.values, label='Log-Normal AFT', color='red')


# Add survival function for Generalized Gamma Regression model
gen_gamma_sf = gen_gamma_reg.predict_survival_function(df_reduced.drop(columns=['tenure_scaled', 'churn'])).mean(axis=1)
plt.step(gen_gamma_sf.index, gen_gamma_sf.values, where='post', label='Generalized Gamma', color='purple')

# Add survival function for Piecewise Exponential Regression model
piecewise_exp_sf = piecewise_exp_reg.predict_survival_function(df_model.drop(columns=['tenure', 'churn'])).mean(axis=1)
plt.step(piecewise_exp_sf.index, piecewise_exp_sf.values, where='post', label='Piecewise Exponential', color='orange')

# Continue with labels, grid, legend, etc.
plt.xlabel('Time (tenure)')
plt.ylabel('Survival Probability')
plt.title('Comparison of Survival Curves from AFT Models')
plt.legend()
plt.grid(True)
plt.show()


sig_features = weibull_aft.summary[weibull_aft.summary['p'] < 0.05].index.get_level_values('covariate').unique().tolist()
print("Significant features:", sig_features)


final_features = [feat for feat in sig_features if feat in df_model.columns]
df_final = df_model[final_features + ['tenure', 'churn']]


weibull_final = WeibullAFTFitter()
weibull_final.fit(df_final, duration_col='tenure', event_col='churn')


timeline = np.linspace(0, df_final['tenure'].max(), 100)

# Predict individual survival functions
surv_funcs = weibull_final.predict_survival_function(df_final.drop(columns=['tenure', 'churn']), times=timeline)

expected_lifetime = surv_funcs.sum(axis=0) * (timeline[1] - timeline[0])

# Assume avg revenue per customer per time unit, e.g., avg_revenue = 100
avg_revenue = 100

# Calculate CLV = expected_lifetime * avg_revenue
clv = expected_lifetime * avg_revenue

# Step 6: Append CLV to original data
df_final = df_final.assign(CLV=clv.values)

segment_summary = df_final.groupby(df['region']).CLV.describe()
print(segment_summary)

marital_summary = df_final.groupby(df['marital']).CLV.describe()
print(marital_summary)

custcat_summary = df_final.groupby(df['custcat']).CLV.describe()
print(custcat_summary)

internet_summary = df_final.groupby(df['internet']).CLV.describe()
print(internet_summary)

