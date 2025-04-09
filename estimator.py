import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pymc as pm
import matplotlib.pyplot as plt

# Define PK models
def one_compartment_model(params, t):
    dose, k, V = params
    return (dose / V) * np.exp(-k * t)

def two_compartment_model(params, t):
    dose, k10, k12, k21, V1 = params
    C1 = (dose / V1) * (k21 / (k21 - k12)) * (np.exp(-k12 * t) - np.exp(-k21 * t))
    return C1

def michaelis_menten_model(params, t):
    dose, Vmax, Km = params
    return (Vmax * dose) / (Km + dose * t)

# Objective function for optimization
def objective_function(params, model, t, observed_data):
    predicted_data = model(params, t)
    return np.sum((predicted_data - observed_data) ** 2)

# Estimate parameters
def estimate_parameters(model, t, observed_data, initial_guess):
    result = minimize(objective_function, initial_guess, args=(model, t, observed_data))
    return result.x

# Calculate AIC
def calculate_aic(model, params, t, observed_data):
    predicted_data = model(params, t)
    residuals = observed_data - predicted_data
    ssr = np.sum(residuals ** 2)
    n = len(observed_data)
    k = len(params)
    aic = n * np.log(ssr / n) + 2 * k
    return aic

# Select best model
def select_best_model(models, t, observed_data, initial_guesses):
    best_aic = float('inf')
    best_model = None
    best_params = None

    for model, initial_guess in zip(models, initial_guesses):
        params = estimate_parameters(model, t, observed_data, initial_guess)
        aic = calculate_aic(model, params, t, observed_data)
        if aic < best_aic:
            best_aic = aic
            best_model = model
            best_params = params

    return best_model, best_params

# Bayesian inference with PyMC3
def bayesian_inference(model, t, observed_data, initial_params):
    with pm.Model() as pk_model:
        dose = pm.Normal('dose', mu=initial_params[0], sigma=1)
        if model == one_compartment_model:
            k = pm.Normal('k', mu=initial_params[1], sigma=1)
            V = pm.Normal('V', mu=initial_params[2], sigma=1)
            params = [dose, k, V]
        elif model == two_compartment_model:
            k10 = pm.Normal('k10', mu=initial_params[1], sigma=1)
            k12 = pm.Normal('k12', mu=initial_params[2], sigma=1)
            k21 = pm.Normal('k21', mu=initial_params[3], sigma=1)
            V1 = pm.Normal('V1', mu=initial_params[4], sigma=1)
            params = [dose, k10, k12, k21, V1]
        elif model == michaelis_menten_model:
            Vmax = pm.Normal('Vmax', mu=initial_params[1], sigma=1)
            Km = pm.Normal('Km', mu=initial_params[2], sigma=1)
            params = [dose, Vmax, Km]

        predicted_data = model(params, t)
        observed = pm.Normal('observed', mu=predicted_data, sigma=1, observed=observed_data)

        trace = pm.sample(1000, return_inferencedata=True)

    return trace

# Main Streamlit app
st.title("Pharmacokinetic Model Selection and Parameter Estimation")

st.markdown("""
## Upload your data
Please upload a CSV file with at least the following columns:
- `time`: Time points
- `concentration`: Observed concentrations
- `covariates`: Any additional covariates for the model
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.write(data.head())

    # Prepare data
    X = data[['time']].values
    y = data['concentration'].values

    # Initial guesses for parameters
    initial_guesses = [
        [1, 1, 1],  # One-compartment model
        [1, 1, 1, 1, 1],  # Two-compartment model
        [1, 1, 1]  # Michaelis-Menten model
    ]

    # Models list
    models = [one_compartment_model, two_compartment_model, michaelis_menten_model]

    # Select the best model
    best_model, best_params = select_best_model(models, X.flatten(), y, initial_guesses)
    st.write(f"Best model: {best_model.__name__} with parameters: {best_params}")

    # Perform Bayesian inference on the best model
    trace = bayesian_inference(best_model, X.flatten(), y, best_params)
    st.write("Bayesian inference completed.")

    # Plot results
    st.write("Predicted vs Observed Concentrations")
    plt.figure()
    plt.scatter(X, y, label="Observed")
    predicted_data = best_model(best_params, X.flatten())
    plt.plot(X, predicted_data, label="Predicted", color='red')
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend()
    st.pyplot(plt)

    # Random Forest for covariate selection
    covariates = data.drop(columns=['time', 'concentration']).values
    rf = RandomForestRegressor()
    rf.fit(covariates, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    st.write("Feature importances:")
    for idx in indices:
        st.write(f"{data.columns[idx+2]}: {importances[idx]:.4f}")

    # Display the formula for the best PK model
    st.write(f"Formula for the best PK model ({best_model.__name__}):")
    if best_model == one_compartment_model:
        st.latex(r"C(t) = \frac{{Dose}}{{V}} e^{{-kt}}")
    elif best_model == two_compartment_model:
        st.latex(r"C_1(t) = \frac{{Dose}}{{V_1}} \frac{{k_{21}}}{{k_{21} - k_{12}}} \left(e^{-k_{12} t} - e^{-k_{21} t}\right)")
    elif best_model == michaelis_menten_model:
        st.latex(r"C(t) = \frac{{V_{max} \cdot Dose}}{{K_m + Dose \cdot t}}")

    # Display the estimated parameters and their RSE
    st.write("Estimated parameters and their Relative Standard Error (RSE):")
    for param, val in zip(best_params, trace.posterior.mean().values()):
        st.write(f"{param}: {val.mean():.4f} (RSE: {np.std(val)/np.mean(val):.4%})")
