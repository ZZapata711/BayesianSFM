import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from pymc import math as pm_math
from scipy.stats import halfnorm, skewnorm
from IPython.display import display

class SNHN: 
    """
    SNHN class to simulate and compare Bayesian Skew-Normal and Normal models 
    for a simple inefficiency estimation problem using PyMC.

    Attributes:
        alpha (float): Intercept of the linear predictor.
        beta (float): Slope coefficient for predictor X.
        sigma_u (float): Standard deviation for inefficiency term.
        sigma_v (float): Standard deviation for noise.
        lambda_skew (float): Skewness parameter for skew-normal model.
        n (int): Number of simulated data points.
        seed (int): Random seed for reproducibility.
    """    
    
    def __init__(self, alpha=5, beta=2, sigma_u=1, sigma_v=1.5, lambda_skew=-0.5, n=50, seed=None):     
        """
        Initializes the model by simulating data and fitting both skew-normal 
        and normal Bayesian models using PyMC.

        Args:
            alpha (float): Intercept term.
            beta (float): Coefficient on predictor X.
            sigma_u (float): Std deviation of inefficiency term.
            sigma_v (float): Std deviation of noise.
            lambda_skew (float): Skewness parameter.
            n (int): Sample size.
            seed (int, optional): Random seed for reproducibility.
        """       
        # Set random seed (Needed)
        if seed:
            self.seed = seed
            np.random.seed(self.seed)
        else:
            self.seed = np.random.randint(1, 10000)
            np.random.seed(self.seed)

        # Parameters
        self.alpha = alpha
        self.beta = beta
        self.sigma_u = sigma_u
        self.sigma_v = sigma_v
        self.lambda_skew = lambda_skew
        self.n = n

        print('---------------------------------------------------\n')
        print("Initializing model generation with parameters: \n")
        print(f"Alpha: {alpha}\n")
        print(f'Beta: {beta}\n')
        print(f'Sigma_u: {sigma_u}\n')
        print(f'Sigma_v: {sigma_v}\n')
        print(f'Lambda: {lambda_skew}\n')
        print(f'Sample size: {n}\n')
        print(f'Random seed: {self.seed}\n')
        
        print('---------------------------------------------------\n')
        # Data Simulation
        X = np.random.normal(1, 1, self.n)

        #Inefficiency term U ~ HalfNormal(0, sigma_u)
        U = halfnorm(scale=self.sigma_u).rvs(self.n)
        
        # Model Generation: Skew Normal
        
        # Location ξ = α + βX − U
        xi = self.alpha + self.beta * X - U

        # Simulate Y ~ SkewNormal(ξ, sigma_v, lambda_skew)
        Y = skewnorm(a=self.lambda_skew, loc=xi, scale=self.sigma_v).rvs(self.n)

        # Put in DataFrame for convenience
        data = pd.DataFrame({"X": X, "Y": Y})

        # Generate SkewNormal Model
        with pm.Model() as model:
            # Priors for parameters
            alpha_ = pm.Normal('alpha', mu=self.alpha, sigma=self.alpha)
            beta_ = pm.Normal('beta', mu=self.beta, sigma=self.beta)
            sigma_u = pm.InverseGamma('sigma_u', alpha=4, beta=3)
            sigma_v = pm.InverseGamma('sigma_v', alpha=2.33, beta=3)
            lam = pm.TruncatedNormal('lam', mu=self.lambda_skew, sigma=1, lower = np.round(self.lambda_skew-4), upper=0)

            # Latent inefficiency U ~ HalfNormal(0, sigma_u)
            U = pm.HalfNormal('U', sigma=sigma_u, shape=self.n)
            self.U = U

            # Linear predictor with inefficiency
            mu = alpha_ + beta_ * data['X'] - U

            # Skew-Normal likelihood
            Y_obs = pm.SkewNormal('Y_obs', mu=mu, sigma=sigma_v, alpha=lam, observed=data['Y'])

            # MCMC Sampling 
            self.skew_trace = pm.sample(6000, tune=2000, target_accept=0.99, return_inferencedata=True, compute_convergence_checks=True, idata_kwargs={'log_likelihood': True})
            self.skew_model = model

        # Generate SkewNormal Model
        with pm.Model() as model:
            # Priors for parameters
            alpha_ = pm.Normal('alpha', mu=self.alpha, sigma=self.alpha)
            beta_ = pm.Normal('beta', mu=self.beta, sigma=self.beta)
            sigma_u = pm.InverseGamma('sigma_u', alpha=4, beta=3)
            sigma_v = pm.InverseGamma('sigma_v', alpha=2.33, beta=3)

            # Latent inefficiency U ~ HalfNormal(0, sigma_u)
            U = pm.HalfNormal('U', sigma=sigma_u, shape=self.n)

            # Linear predictor with inefficiency
            mu = alpha_ + beta_ * data['X'] - U

            # Skew-Normal likelihood
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma_v, observed=data['Y'])

            # MCMC Sampling 
            self.norm_trace = pm.sample(6000, tune=2000, target_accept=0.99, return_inferencedata=True, compute_convergence_checks=True, idata_kwargs={'log_likelihood': True})
            self.norm_model = model

        
    def plot_trace(self, model_type='skew'):
        """
        Plots trace plots for MCMC diagnostics.

        Args:
            model_type (str): One of {'skew', 'norm'}. Determines which model's trace to plot.
        """    
        trace = self.skew_trace if model_type == 'skew' else self.norm_trace
        az.plot_trace(trace, figsize=(10,20))
        plt.suptitle(f'Trace Plot for {model_type.upper()} Model')
        plt.show()

    def plot_posterior(self, model_type='skew'):
        """
        Plots posterior distributions of the model parameters.

        Args:
            model_type (str): One of {'skew', 'norm'}. Determines which model's posterior to plot.
        """        
        trace = self.skew_trace if model_type == 'skew' else self.norm_trace
        az.plot_posterior(trace)
        plt.suptitle(f'Posterior Distributions for {model_type.upper()} Model')
        plt.show()

    def compare_loo(self):
        """
        Computes and displays Leave-One-Out (LOO) cross-validation comparison between
        the skew-normal and normal models using ArviZ.
        """
        comparison = az.compare({'Skew Normal': self.skew_trace, 'Normal': self.norm_trace})
        display(comparison)

    def compare_rmse(self, choice='Predictions'):
        """
        Computes and prints RMSE comparisons for predictions, parameters, or inefficiency terms.

        Args:
            choice (str): One of {'Predictions', 'Parameters', 'Inefficiency'}.
                - 'Predictions': compares predicted vs. observed Y.
                - 'Parameters': compares posterior means to true parameter values.
                - 'Inefficiency': compares estimated inefficiency U to true simulated U.
        """        
        # Y_Obs RMSE
        if choice == 'Predictions':
            # Skew Normal Model
            with self.skew_model:
                ppc_skew = pm.sample_posterior_predictive(self.skew_trace, var_names=['Y_obs'])
            y_pred_skew = ppc_skew.posterior_predictive['Y_obs'].mean(dim=['chain', 'draw']).values
            y_obs_skew = self.skew_trace.observed_data['Y_obs'].values
            rmse_skew = np.sqrt(np.mean((y_obs_skew - y_pred_skew) ** 2))
            
            with self.norm_model:
                ppc_norm = pm.sample_posterior_predictive(self.norm_trace, var_names=['Y_obs'])
            y_pred_norm = ppc_norm.posterior_predictive['Y_obs'].mean(dim=['chain', 'draw']).values
            y_obs_norm = self.norm_trace.observed_data['Y_obs'].values
            rmse_norm = np.sqrt(np.mean((y_obs_norm - y_pred_norm) ** 2))

            df = pd.DataFrame({
                'n': [self.n],
                'Skew Model RMSE': [rmse_skew],
                'Normal Model RMSE': [rmse_norm]
            })

            print(df)
            # If collecting results, uncomment the return statement
            # return df 

        # Parameter Estimate RMSE
        if choice == "Parameters":

            true_vals_skew = np.array([self.alpha, self.beta, self.sigma_v, self.sigma_u, self.lambda_skew])
            true_vals_norm = np.array([self.alpha, self.beta, self.sigma_v, self.sigma_u])

            skew_summary = az.summary(self.skew_trace, var_names=['alpha', 'beta', 'sigma_v', 'sigma_u', 'lam'])
            norm_summary = az.summary(self.norm_trace, var_names=['alpha', 'beta', 'sigma_v', 'sigma_u'])

            skew_means = skew_summary['mean'].values
            norm_means = norm_summary['mean'].values

            rmse_skew = np.sqrt((skew_means - true_vals_skew)**2)
            rmse_norm = np.sqrt((norm_means - true_vals_norm)**2)

            df = pd.DataFrame({
                "Parameter": ['alpha', 'beta', 'sigma_v', 'sigma_u', 'lambda'],
                'Skew Model RMSE': list(rmse_skew),
                'Normal Model RMSE': list(rmse_norm) + [np.nan]
            })
            
            print(df)
            # If collecting results, uncomment the return statement
            # return df             

        if choice == "Inefficiency":
            np.random.seed(self.seed)
            U_true = halfnorm(scale=1).rvs(self.n)
            # Skew Normal Model
            U_est_skew = self.skew_trace.posterior['U'].mean(dim=['chain', 'draw']).values
            rmse_u_skew = np.sqrt(np.mean((U_true - U_est_skew)**2))
            
            # Normal Model
            U_est_norm = self.norm_trace.posterior['U'].mean(dim=['chain', 'draw']).values
            rmse_u_norm = np.sqrt(np.mean((U_true - U_est_norm)**2))

            df = pd.DataFrame({
                'Sample size (n)': [self.n],
                'Skew Model RMSE (U)': [rmse_u_skew],
                'Normal Model RMSE (U)': [rmse_u_norm]
            })
            print(df)
            # If collecting results, uncomment the return statement
            # return df
