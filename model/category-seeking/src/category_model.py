
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import copy

class SimpleCategorySpace:
    def __init__(self, 
                 mu_prior_means=[0,0], 
                 mu_prior_sds = [10,10], 
                 alpha_priors = [2,2], 
                 beta_priors = [2,2]):
        """
        Initialize a simple category space with priors over category means.
        
        Parameters:
        -----------
        mu_prior_means : list
            Prior means for category A and B
        mu_prior_sds : list
            Prior standard deviations for means of category A and B
        alpha_priors, beta_priors : list
            Prior parameters for variance (not used in this simplified version)
        """
        self.mu_prior_means = mu_prior_means
        self.mu_prior_sds = mu_prior_sds
        self.alpha_priors = alpha_priors
        self.beta_priors = beta_priors

        # Initialize posterior parameters to be the same as priors
        self.mu_posterior_means = mu_prior_means
        self.mu_posterior_sds = mu_prior_sds

        # Fixed category variances (simplified model)
        self.cat_a_var = 1
        self.cat_b_var = 1

        # Initialize observation storage
        self.observations = []

        # Initialize step counter
        self.step = 0

        # Setup for visualization
        self.mean_x_values = np.arange(-25, 25, .1)

        # Calculate prior PDFs for visualization
        self.cat_a_prior_mean_pdf = stats.norm.pdf(self.mean_x_values, 
                                              self.mu_prior_means[0], 
                                              self.mu_prior_sds[0])
        self.cat_b_prior_mean_pdf = stats.norm.pdf(self.mean_x_values, 
                                              self.mu_prior_means[1], 
                                              self.mu_prior_sds[1])
        
        self.cat_a_posterior_mean_pdf = self.cat_a_prior_mean_pdf.copy()
        self.cat_b_posterior_mean_pdf = self.cat_b_prior_mean_pdf.copy()


    def set_true_categories(self, true_mu_a=-3, true_mu_b=3, true_sigma=1):
        """
        Set the true category parameters for generating observations.
        
        Parameters:
        -----------
        true_mu_a, true_mu_b : float
            True means for category A and B
        true_sigma : float
            True standard deviation for both categories
        """
        
        self.true_mu_a = true_mu_a
        self.true_mu_b = true_mu_b
        self.true_sigma = true_sigma
        
        # Store for visualization
        self.true_categories = {
            'A': {'mean': true_mu_a, 'sd': true_sigma},
            'B': {'mean': true_mu_b, 'sd': true_sigma}
        }
        
        # Define the category boundary
        self.boundary_region = (-1, 1)
        
        return self

    def get_category_probabilities(self, x_value):
        """
        Calculate the probability of each category label for a given x value
        based on the true underlying distributions
        """
        # Calculate likelihood of x under each category distribution
        likelihood_a = stats.norm.pdf(x_value, self.true_mu_a, self.true_sigma)
        likelihood_b = stats.norm.pdf(x_value, self.true_mu_b, self.true_sigma)
        
        # Normalize to get probabilities
        total = likelihood_a + likelihood_b
        p_cat_a = likelihood_a / total
        p_cat_b = likelihood_b / total
        
        return p_cat_a, p_cat_b

    def update_posterior(self, observation, category):
        """
        Update the posterior distribution after observing a data point.
        
        Parameters:
        -----------
        observation : float
            The observed data point
        category : int
            The category label (0 for category A, 1 for category B)
        """

        self.step += 1
        
        # Store the observation
        self.observations.append((observation, category))

        # Update the posterior for the observed category
        # For normal with known variance (1), the posterior is also normal
        # New mean = (prior_precision * prior_mean + n * sample_mean) / (prior_precision + n)
        # New precision = prior_precision + n
        
        # Prior precision = 1 / variance = 1 / (standard_deviation^2)
        prior_precision = 1 / (self.mu_posterior_sds[category] ** 2)
        # Likelihood precision for n=1 observation with variance=1 is just 1
        likelihood_precision = 1
        
        # Update mean
        posterior_precision = prior_precision + likelihood_precision
        posterior_mean = (prior_precision * self.mu_posterior_means[category] + 
                        observation) / posterior_precision
        
        # Update standard deviation
        posterior_sd = np.sqrt(1 / posterior_precision)
        
        # Store updated parameters
        self.mu_posterior_means[category] = posterior_mean
        self.mu_posterior_sds[category] = posterior_sd
        
        # Update the PDFs for visualization
        self.cat_a_posterior_mean_pdf = stats.norm.pdf(self.mean_x_values, 
                                                self.mu_posterior_means[0], 
                                                self.mu_posterior_sds[0])
        self.cat_b_posterior_mean_pdf = stats.norm.pdf(self.mean_x_values, 
                                                self.mu_posterior_means[1], 
                                                self.mu_posterior_sds[1])
        
        return self

    def calculate_entropy(self, category=None):
        """
        Calculate the entropy of the current belief distributions.
        
        For normal distributions, the differential entropy is:
        0.5 * log(2 * pi * e * sigma^2)
        
        Parameters:
        -----------
        category : int or None
            If provided (0 or 1), calculates entropy for just that category.
            If None, calculates total entropy for both categories.
        
        Returns:
        --------
        float
            Entropy of the distributions
        """
        # Use posterior means and SDs
        sd_a = self.mu_posterior_sds[0]
        sd_b = self.mu_posterior_sds[1]
        
        # Calculate differential entropy for normal distributions
        entropy_a = 0.5 * np.log(2 * np.pi * np.e * sd_a**2)
        entropy_b = 0.5 * np.log(2 * np.pi * np.e * sd_b**2)
        
        # Return based on category parameter
        if category == 0:
            return entropy_a
        elif category == 1:
            return entropy_b
        else:
            # Total entropy is the sum
            return entropy_a + entropy_b

    def calculate_eig(self, potential_x, category=None):
        """
        Calculate the Expected Information Gain for a potential observation at x.
        
        Parameters:
        -----------
        potential_x : float
            The feature value to calculate EIG for
        category : int or None
            If provided (0 or 1), calculates EIG for just that category.
            If None, calculates total EIG for both categories.
            
        Returns:
        --------
        float or dict
            The expected information gain (float if category is provided, 
            dict with 'total', 'cat_a', and 'cat_b' keys otherwise)
        """
        # Calculate current entropy (before observation)
        if category is not None:
            current_entropy = self.calculate_entropy(category)
        else:
            current_entropy_total = self.calculate_entropy()
            current_entropy_a = self.calculate_entropy(0)
            current_entropy_b = self.calculate_entropy(1)
        
        # Get probabilities of each category label for this x
        p_cat_a, p_cat_b = self.get_category_probabilities(potential_x)
        
        # For category A
        model_copy_a = copy.deepcopy(self)
        model_copy_a.update_posterior(potential_x, 0)
        entropy_a_total = model_copy_a.calculate_entropy()
        entropy_a_cat_a = model_copy_a.calculate_entropy(0)
        entropy_a_cat_b = model_copy_a.calculate_entropy(1)
        
        # For category B
        model_copy_b = copy.deepcopy(self)
        model_copy_b.update_posterior(potential_x, 1)
        entropy_b_total = model_copy_b.calculate_entropy()
        entropy_b_cat_a = model_copy_b.calculate_entropy(0)
        entropy_b_cat_b = model_copy_b.calculate_entropy(1)
        
        # Weight entropies by category probabilities
        expected_posterior_entropy_total = p_cat_a * entropy_a_total + p_cat_b * entropy_b_total
        expected_posterior_entropy_a = p_cat_a * entropy_a_cat_a + p_cat_b * entropy_b_cat_a
        expected_posterior_entropy_b = p_cat_a * entropy_a_cat_b + p_cat_b * entropy_b_cat_b
        
        # Calculate EIG
        if category == 0:
            return current_entropy - expected_posterior_entropy_a
        elif category == 1:
            return current_entropy - expected_posterior_entropy_b
        else:
            return {
                'total': current_entropy_total - expected_posterior_entropy_total,
                'cat_a': current_entropy_a - expected_posterior_entropy_a,
                'cat_b': current_entropy_b - expected_posterior_entropy_b
            }

    def calculate_eig_across_range(self, x_range=(-10,10), num_points=100, category=None):
        """
        Calculate EIG across a range of potential x values.
        
        Parameters:
        -----------
        x_range : tuple
            Range of x values to evaluate (defaults to -10 to 10)
        num_points : int
            Number of points to evaluate
        category : int or None
            If provided (0 or 1), calculates EIG for just that category.
            If None, calculates total EIG for both categories.
            
        Returns:
        --------
        dict
            Dictionary with x_values and eig_values (which may be a dict if category=None)
        """
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        
        if category is not None:
            eig_values = np.zeros_like(x_values)
            for i, x in enumerate(x_values):
                eig_values[i] = self.calculate_eig(x, category)
            return {'x_values': x_values, 'eig_values': eig_values}
        else:
            eig_total = np.zeros_like(x_values)
            eig_cat_a = np.zeros_like(x_values)
            eig_cat_b = np.zeros_like(x_values)
            
            for i, x in enumerate(x_values):
                eig_dict = self.calculate_eig(x)
                eig_total[i] = eig_dict['total']
                eig_cat_a[i] = eig_dict['cat_a']
                eig_cat_b[i] = eig_dict['cat_b']
                
            return {
                'x_values': x_values,
                'eig_values': {
                    'total': eig_total,
                    'cat_a': eig_cat_a,
                    'cat_b': eig_cat_b
                }
            }

    def plot_mean_distributions(self, title="Distribution over Category Means"):
        """Visualize the current beliefs about category means"""
        plt.figure(figsize=(10, 6))
        
        # Plot distribution over means for both categories
        plt.plot(self.mean_x_values, self.cat_a_prior_mean_pdf, 'r-', 
                label=f'Category A Mean (μ={self.mu_prior_means[0]:.2f}, σ={self.mu_prior_sds[0]:.2f})')
        plt.plot(self.mean_x_values, self.cat_b_prior_mean_pdf, 'b-', 
                label=f'Category B Mean (μ={self.mu_prior_means[1]:.2f}, σ={self.mu_prior_sds[1]:.2f})')
        
        # If we have posteriors after updating, plot those too
        if hasattr(self, 'cat_a_posterior_mean_pdf'):
            plt.plot(self.mean_x_values, self.cat_a_posterior_mean_pdf, 'r--', 
                    label=f'Category A Posterior (step {self.step})')
            plt.plot(self.mean_x_values, self.cat_b_posterior_mean_pdf, 'b--', 
                    label=f'Category B Posterior (step {self.step})')
        
        plt.title(title)
        plt.xlabel('Mean Value')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_simple_category_distributions(self, show_true=False, feature_range=(-20, 20), num_points=1000):
        """Visualize simplified category distributions using expected means"""
        x = np.linspace(feature_range[0], feature_range[1], num_points)
        
        # Use expected means from current beliefs
        expected_mean_a = self.mu_prior_means[0]
        expected_mean_b = self.mu_prior_means[1]
        
        # If we have updated posteriors, use those means instead
        if hasattr(self, 'mu_posterior_means'):
            expected_mean_a = self.mu_posterior_means[0]
            expected_mean_b = self.mu_posterior_means[1]
        
        # Plot distributions
        plt.figure(figsize=(10, 6))
        plt.plot(x, stats.norm.pdf(x, expected_mean_a, np.sqrt(self.cat_a_var)), 
                'r-', label=f'Category A (μ={expected_mean_a:.2f})')
        plt.plot(x, stats.norm.pdf(x, expected_mean_b, np.sqrt(self.cat_b_var)), 
                'b-', label=f'Category B (μ={expected_mean_b:.2f})')
        
        if show_true and hasattr(self, 'true_categories'):
            plt.plot(x, stats.norm.pdf(x, self.true_categories['A']['mean'], 
                                    self.true_categories['A']['sd']),
                    'r:', linewidth=2, label='True Category A')
            plt.plot(x, stats.norm.pdf(x, self.true_categories['B']['mean'], 
                                    self.true_categories['B']['sd']),
                    'b:', linewidth=2, label='True Category B')
            
            # Shade the ambiguous boundary region
            if hasattr(self, 'boundary_region'):
                plt.axvspan(self.boundary_region[0], self.boundary_region[1], 
                        alpha=0.2, color='gray', label='Ambiguous Region')

        plt.title("Simplified Category Distributions")
        plt.xlabel("Feature Value")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_eig(self, x_range=(-10,10), num_points=100, separate_plots=True):
        """
        Plot the Expected Information Gain across a range of x values.
        
        Parameters:
        -----------
        x_range : tuple
            Range of x values to evaluate
        num_points : int
            Number of points to evaluate
        separate_plots : bool
            If True, creates separate plots for total EIG and per-category EIG
        """
        result = self.calculate_eig_across_range(x_range, num_points)
        x_values = result['x_values']
        
        # Function to add reference lines to the plot
        def add_reference_lines(ax):
            # Mark the posterior means
            ax.axvline(x=self.mu_posterior_means[0], color='r', linestyle='--', 
                      label=f'Category A Mean (μ={self.mu_posterior_means[0]:.2f})')
            ax.axvline(x=self.mu_posterior_means[1], color='b', linestyle='--',
                      label=f'Category B Mean (μ={self.mu_posterior_means[1]:.2f})')
            
            # If true categories are set, mark them too
            if hasattr(self, 'true_categories'):
                ax.axvline(x=self.true_categories['A']['mean'], color='r', linestyle=':',
                          label=f'True Category A Mean (μ={self.true_categories["A"]["mean"]:.2f})')
                ax.axvline(x=self.true_categories['B']['mean'], color='b', linestyle=':',
                          label=f'True Category B Mean (μ={self.true_categories["B"]["mean"]:.2f})')
                
                # Shade the ambiguous boundary region
                if hasattr(self, 'boundary_region'):
                    ax.axvspan(self.boundary_region[0], self.boundary_region[1], 
                            alpha=0.2, color='gray', label='Ambiguous Region')
        
        if separate_plots:
            # Plot total EIG and per-category EIG in separate subplots
            fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
            
            # Total EIG plot
            ax = axes[0]
            ax.plot(x_values, result['eig_values']['total'], 'g-', linewidth=2, label='Total EIG')
            add_reference_lines(ax)
            ax.set_title('Total Expected Information Gain')
            ax.set_ylabel('EIG')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Category A EIG plot
            ax = axes[1]
            ax.plot(x_values, result['eig_values']['cat_a'], 'r-', linewidth=2, label='Category A EIG')
            add_reference_lines(ax)
            ax.set_title('Expected Information Gain for Category A')
            ax.set_ylabel('EIG')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Category B EIG plot
            ax = axes[2]
            ax.plot(x_values, result['eig_values']['cat_b'], 'b-', linewidth=2, label='Category B EIG')
            add_reference_lines(ax)
            ax.set_title('Expected Information Gain for Category B')
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('EIG')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        else:
            # Plot all on one axis
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, result['eig_values']['total'], 'g-', linewidth=2, label='Total EIG')
            plt.plot(x_values, result['eig_values']['cat_a'], 'r-', linewidth=2, label='Category A EIG')
            plt.plot(x_values, result['eig_values']['cat_b'], 'b-', linewidth=2, label='Category B EIG')
            
            # Add reference lines
            add_reference_lines(plt.gca())
            
            plt.title('Expected Information Gain Across Feature Space')
            plt.xlabel('Feature Value')
            plt.ylabel('Expected Information Gain')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return result
