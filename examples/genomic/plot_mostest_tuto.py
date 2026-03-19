"""
MOSTest
=======

Eamplify MOSTest approach which performs a multi-phenotype GWAS.

The method was first introduced in this paper
from [van der Meer, Nat. Comm.,2020](https://doi.org/10.1038/s41467-020-17368-1).
The current example consider the situation where the phenotypes are linearly
dependent.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from scipy.stats import norm, gamma

np.set_printoptions(suppress=False, precision=2)


# %%
# Building the dataset
# --------------------
#
# The multivariate phenotype
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First, a multi variate phenotype is defined. Each trait is supposed to
# characterize a different sub-populations within the sample.
#
# - The number of traits is :math:`n\_traits` traits
# - The number of samples (subjects) is :math:`m`.
#
# For example, each trait of the multivariate phenotype represents a
# dimension of a latent space,  which are supposed to encode information for
# one specific part of the global population. For each sub-population is
# generated the data (matching in fact one dimension).
# For the first sub-population, the subjects have a value attributed. Those
# values follow a normal distribution.
# The subjects are sorted by increasing values. Then, all the other subjects
# have a 0 value.
# The same process is repeated for all the sub-populations.
#
# For our simulation, and for each given trait, we will arbitrarily draw some
# specific samples according to a distribution and will zero the other
# samples. Without loss of generality we will consider the :math:`m_1`
# first samples of trait :math:`X_1` (and zero the others), then we will
# consider the :math:`m_2` samples following the :math:`m_1` ones of trait
# :math:`X_2` and zero the other samples, etc.
#
# Then, a certain amount of noise is added.

# dimensions of the simulated data
m = 5000        # sample size
n_traits = 4    # number of uncorrelated traits
trait_names = [f'X{i}' for i in range(1,n_traits+1)]
trait_names


# %%
# Set the size of the different sub-population of subjects in each trait

m1 = 2000
m2 = 500
m3 = 1000
m4 = m - m1 - m2 - m3


# %%
# Now let's generate the phenotypes

def generate_traits(trait_runlengthes, list_sigma=None, law='normal'):
    """
    Generate traits based on a given distribution law.
    Parameters:
    -----------
        trait_runlengthes: numpy array of integers, 
            each representing the number of non zero samples for each trait.
        list_sigma: list of floats,
            standard deviations for each trait.
        law: string, 
            distribution law to use ('normal' or 'gamma').
    Returns:
    --------
        mul_phenotype: 2D numpy array, 
            each column containing the generated trait data.
    """
    trait_runlengthes = np.array(trait_runlengthes)
    if list_sigma == None:
        list_sigma = [1 for i in range(len(trait_runlengthes))]

    list_X = []
    if law == 'normal':
        for i in range(len(trait_runlengthes)):
            X = np.concatenate([
                    np.zeros(trait_runlengthes[0:i].sum()), 
                    np.sort(np.random.normal(loc=0, scale=list_sigma[i], size=trait_runlengthes[i])), 
                    np.zeros(trait_runlengthes.sum()-trait_runlengthes[0:i+1].sum())
                    ])
            list_X.append(X)
    elif law == 'gamma':
        for i in range(len(trait_runlengthes)):
            X = np.concatenate([
                    np.zeros(trait_runlengthes[0:i].sum()), 
                    np.sort(gamma.rvs(a=2, scale=list_sigma[i], size=trait_runlengthes[i])), 
                    np.zeros(trait_runlengthes.sum()-trait_runlengthes[0:i+1].sum())
                    ])
            list_X.append(X)
    else:
        raise ValueError("Unsupported distribution law. Use 'normal' or 'gamma'.")
    mul_phenotype = np.array(list_X).T
    return mul_phenotype


mul_phenotype_latent = generate_traits(np.array([m1, m2, m3, m4]))

# we keep the "latent" for future plots
mul_phenotype = mul_phenotype_latent.copy()

# Mean and Variance
print('Mean for each column:', mul_phenotype.mean(axis=0))
print('Variance for each column:', mul_phenotype.var(axis=0))
fig, ax = plt.subplots(figsize=(12, 3))
c = ax.pcolor(mul_phenotype, cmap="jet")  # or "viridis", "plasma", etc.
# Set ticks in the middle of each cell
ax.set_xticks(np.arange(0.5, mul_phenotype.shape[1], 1))

# Set tick labels
ax.set_xticklabels(trait_names)
fig.colorbar(c, ax=ax)


# %%
# Let us add noise Sparse representation space with noise

noise = True
if noise:
    for i in range(len(trait_names)):
        mul_phenotype[:,i] += np.random.randn(m)*4
fig, ax = plt.subplots(figsize=(12, 3))
c = ax.pcolor(mul_phenotype, cmap="jet")  # or "viridis", "plasma", etc.
# Set ticks in the middle of each cell
ax.set_xticks(np.arange(0.5, mul_phenotype.shape[1], 1))

# Set tick labels
ax.set_xticklabels(trait_names)
fig.colorbar(c, ax=ax)


# %%
# Let's now add a duplicate of two columns.

n_traits += 2
trait_names = [f'X{i}' for i in range(1,n_traits+1)]
print(trait_names)
mul_phenotype = np.hstack((mul_phenotype, np.tile(mul_phenotype[:, [-1]], 1)))
mul_phenotype = np.hstack((mul_phenotype, np.tile(mul_phenotype[:, [0]], 1)))
mul_phenotype[:,-1] += np.random.randn(m)*2
mul_phenotype[:,-2] += np.random.randn(m)*2


# %%
# Mean and Variance
#

print(f'Mean for each column before scaling:', mul_phenotype.mean(axis=0))
print(f'Variance for each column before scaling:', mul_phenotype.var(axis=0), '\n')

scaler = StandardScaler()
mul_phenotype = scaler.fit_transform(mul_phenotype)

# Mean and Variance
print(f'Mean for each column before scaling:', mul_phenotype.mean(axis=0))
print('Variance for each column before scaling:', mul_phenotype.var(axis=0))
fig, ax = plt.subplots(figsize=(12, 3))
c = ax.pcolor(mul_phenotype, cmap="jet")  # or "viridis", "plasma", etc.
# Set ticks in the middle of each cell
ax.set_xticks(np.arange(0.5, mul_phenotype.shape[1], 1))

# Set tick labels
ax.set_xticklabels(trait_names)
fig.colorbar(c, ax=ax)


# %%
# Definition of the genotype
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's define the genotype (one variant). The linspace will naturally
# corralate with the sorted traits.
# The genotype is defined to match the normal distribution, ie the subjects
# with a low value have a 0 as genotype, 
# the subjects in the middle have a 1 and the subjects with a high value
# have a 2.
# The genotype correspond to the number of minor allele for a given fake SNP.

genotype = np.concatenate([np.linspace(0, 2, m1), np.linspace(0, 2, m2), np.linspace(0, 2, m3), np.linspace(0, 2, m4)])
genotype = genotype.round()


# %%
# Let's plot the genotype and the (latent) multivariate phenotype

# two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 3))
axs[0].scatter(genotype,range(0,m))
c = axs[1].pcolor(mul_phenotype_latent, cmap="jet")
axs[1].set_xticks(np.arange(0.5, mul_phenotype_latent.shape[1], 1))
axs[1].set_xticklabels(trait_names[:-2])
fig.colorbar(c, ax=axs[1])


# %%
# A few description plots on the simulated data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Correlation between the different dimensions

corr = np.corrcoef(mul_phenotype, rowvar=False)
sns.heatmap(corr, cmap="YlOrBr", annot=True, fmt=".2f")


# %%
# Lets study the correlation between the different traits of the 
# mul_phenotype and with the genotype one the one hand or a permuted genotype
# on the other hand

# Consider a pandas dataframe for better handling
mpheno_geno = pd.DataFrame(mul_phenotype, columns=[f'X{i}' for i in range(1,n_traits+1)])
mpheno_geno['Genotype'] = genotype
mpheno_geno['PermGenotype'] = np.random.permutation(genotype)
mpheno_geno
correlations = mpheno_geno.corr()
correlations[['Genotype','PermGenotype']].drop(['Genotype', 'PermGenotype'])


# %%
# The MOSTest principles
# ----------------------
#
# Compute the classical association metrics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Also referred as "Univariate GWAS procedure" in [Van der Meer et al., 2022]
#
# In order to obtain t-val, p-val of association of the genotype for each
# trait of the mul_phenotype a univariate regression is performed. The data
# frame mpheno_geno is used.
#
# Can be seen from the results that the genotype predicts each dimension.

def UniVar_reg(mpheno_geno, traits):
    geno = sm.add_constant(mpheno_geno['Genotype'])
    list_betas_orig = []
    list_z_score_orig = []

    pd_list = []
    for trait in traits:
        model_orig = sm.OLS(mpheno_geno[trait], geno)
        results_orig = model_orig.fit()
        pd_list.append(pd.DataFrame(
                    {'Trait': [trait],
                    'Beta': [results_orig.params.iloc[1]],
                    't-value': [results_orig.tvalues.iloc[1]],
                    'p-value': [results_orig.pvalues.iloc[1]]}
                    )
        )

        list_betas_orig.append(results_orig.params.iloc[1])
        list_z_score_orig.append(results_orig.tvalues.iloc[1])

    results = pd.concat(pd_list, ignore_index=True)
    results.set_index('Trait', inplace=True)
    return results, list_betas_orig, list_z_score_orig

univariates_gwases, list_betas_orig, list_z_score_orig = UniVar_reg(mpheno_geno, trait_names)
univariates_gwases


# %%
# The Mahalanobis norm for $H_0$
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The following function compute the Mahalanobis norm considering permuted
# genotype to assess the association under the null.
#
# Retaining the notation of the Van der Meer's paper, we define the
# mahalanobis_norm_perm. The parameters are the mul_phenotype, the genotype
# (here the mpheno_geno parameter). The nb_perm_pheno is to specify the number
# of permutation. The :math:`R` parameter is the correlation matrix of the
# paper. Please note that :math:`R` is NOT computed the same way as in the
# paper.

from tqdm.auto import tqdm

# the mahalanobis_norm_perm function
#
def mahalanobis_norm_perm(mpheno_geno, R, nb_perm_geno = 1):
    """
    Input:
        mpheno_geno: pandas dataframe
            contains the traits phenotype and the genotype data ('Genotype' column).
        R: numpy array,
            correlation matrix of the traits in mpheno_geno.
        nb_perm_geno: int
    
    Output:
        list_mahalanobis_norm_perm: list of float
    """
    list_mahalanobis_norm_perm = []
    for i in tqdm(range(nb_perm_geno),'Permutation'):
        mpheno_geno['PermGenotype'] = np.random.permutation(mpheno_geno['Genotype'])
        perm_geno = sm.add_constant(mpheno_geno['PermGenotype'])
        list_z_score_perm = []

        traits = mpheno_geno.drop(columns=["Genotype", "PermGenotype"]).columns.tolist()
        for trait in traits:
            model_perm = sm.OLS(mpheno_geno[trait], perm_geno)
            results_perm = model_perm.fit()
            list_z_score_perm.append(results_perm.tvalues.iloc[1])

        z_score_perm = np.array(list_z_score_perm)
        mahalanobis_norm_perm = (z_score_perm @ np.linalg.inv(R)) @ z_score_perm.T

        list_mahalanobis_norm_perm.append(mahalanobis_norm_perm)
    return list_mahalanobis_norm_perm


nb_perm_geno = 2
list_mahalanobis_norm_perm = mahalanobis_norm_perm(mpheno_geno, corr, nb_perm_geno)
list_mahalanobis_norm_perm


# %%
# The MOSTest idea
# ~~~~~~~~~~~~~~~~
#
# The terms of the list list_mahalanobis_norm_perm contain values of the 
# statistics :math:`X_k^2` of the paper.
# They follow the distribution the two parameter distribution
# :math:`gamma(\alpha, loc)`.
#
# Let's fit this gamma function.

def fit_gamma(list_mahalanobis_norm_perm, plot_fit=False):
    """
    Fit a gamma distribution to the list of Mahalanobis norms.
    
    Parameters:
    -----------
        list_mahalanobis_norm_perm: list of float,
            Mahalanobis norms from permutations.
        plot_fit: bool, optional,
            whether to plot the fitted gamma distribution against the data.
    
    Returns:
    --------
        fitted_params: tuple,
            parameters of the fitted gamma distribution (fit_alpha, fit_loc, fit_scale).
    """
    fit_alpha, fit_loc, fit_scale = gamma.fit(list_mahalanobis_norm_perm)
    nb_perm_geno = len(list_mahalanobis_norm_perm)

    if plot_fit:
        plt.hist(gamma.rvs(a=fit_alpha, loc=fit_loc, scale=fit_scale, size=nb_perm_geno), alpha=0.5, bins=60)
        plt.hist(list_mahalanobis_norm_perm, alpha=0.5, bins=60)

    return fit_alpha, fit_loc, fit_scale

# Generate a list of Mahalanobis norms from permutations
list_mahalanobis_norm_perm = mahalanobis_norm_perm(mpheno_geno, corr, 5000)
print('Distributions: blue is theoretical gamma and orange is the empirical distribution of the Mahalanobis norms from permutations')
alpha, loc, scale = fit_gamma(list_mahalanobis_norm_perm, plot_fit=True)


# %%
# Then the MOSTest p-val is based on:
# - the observed "Mahalanobis combination" of the classical associations with
#   the traits of the mul_phenotype (and using correctly ordered phenotype)
# - the cdf of the distribution of the "Mahalanobis combination" built
#   empirically using permuted genoype.
#
# Knowing this cdf is a gamma function, the p-val is the integral of the tail
# of the cdf higher than the observed value.

def MOSTest(z_score_observed, R, fit_alpha, fit_loc, fit_scale):
    """ Perform the MOSTest using the original z-scores and the fitted gamma distribution.
    Parameters: 
    -----------
        z_score_observed: list of float,
            original z-scores from the univariate GWAS.
        R: numpy array,
            correlation matrix of the traits.
        fit_alpha: float,
            shape parameter of the fitted gamma distribution.
        fit_loc: float,
            location parameter of the fitted gamma distribution.
        fit_scale: float,
            scale parameter of the fitted gamma distribution.
    Returns:
    --------
        p_value_orig: float,
            p-value from the MOSTest.
    """
    z_score_orig = z_score_observed

    print(f'Observed associations: {z_score_orig}')
    mahalanobis_norm_orig = (z_score_orig @ np.linalg.inv(R)) @ z_score_orig.T
    print(f'Mahalanobis norm on obseved associations: {mahalanobis_norm_orig}')

    p_value_orig = 1-gamma.cdf(mahalanobis_norm_orig, a=fit_alpha, loc=fit_loc, scale=fit_scale)
    print(f'MOSTEST p-value: {p_value_orig}')

    return p_value_orig


# Perform the MOSTest
z_score_observed = univariates_gwases['t-value'].values
p_val = MOSTest(z_score_observed, corr, alpha, loc, scale)


# %%
# Comparison of the methods
# -------------------------
#
# Univariate regression: Each dimension predicts the genotype
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

list_betas = []
pd_list = []

for trait in [f'X{i}' for i in range(1,n_traits+1)]:
    dim = sm.add_constant(mpheno_geno[trait])
    model = sm.OLS(mpheno_geno['Genotype'],dim)
    results = model.fit()
    pd_list.append(pd.DataFrame(
                    {'Trait': [trait],
                    'Beta': [results.params.iloc[1]],
                    't-value': [results.tvalues.iloc[1]],
                    'p-value': [results.pvalues.iloc[1]]}
                    ))
univariate = pd.concat(pd_list, ignore_index=True)
univariate.set_index('Trait', inplace=True)

univariate


# %%
# Multivariate regression: set of dimension to predict the genotype (linear)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Multivariate regression: set of dimension to predict the genotype (linear)
#

pheno = sm.add_constant(mpheno_geno[[f'X{i}' for i in range(1,n_traits+1)]])
model = sm.OLS(mpheno_geno['Genotype'], pheno)
results = model.fit()
#print("Estimated betas:", '\n', results.params, '\n') # to get the betas
#print("t_values:", '\n', results.tvalues, '\n') # which is in fact also the z-score
#print("P-values:", '\n', results.pvalues, '\n')


np.array(results.params[[f'X{i}' for i in range(1,n_traits+1)]].to_list())/results.params['X1']

multivariate = pd.DataFrame({'Beta':results.params[['const']+[f'X{i}' for i in range(1,n_traits+1)]],
                                't-values': results.tvalues[['const']+[f'X{i}' for i in range(1,n_traits+1)]],
                                'p-values': results.pvalues[['const']+[f'X{i}' for i in range(1,n_traits+1)]]})
#multivariate.set_index(multivariate.index.str.replace('X', 'Trait '), inplace=True)
multivariate


# %%
# Multivariate regression: set of dimension to predict the genotype (CCA)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The coefficients of the linear model such that Y is approximated as 
# :math:`Y = X @ coef\_^T + intercept\_.`

cca = CCA(n_components=1)
cca.fit(pheno, mpheno_geno['Genotype'])
#print(cca.coef_.shape)
#print(np.array(cca.coef_[0, 1:])/cca.coef_[0, 1]) # The coefficients of the linear model such that Y is approximated as Y = X @ coef_.T + intercept_.

coef_cca = pd.DataFrame({'normalized loadings':list(np.array(cca.coef_[0, 1:])/cca.coef_[0, 1])})
coef_cca.index = [f'X{i}' for i in range(1,n_traits+1)]
coef_cca.index.name = 'Trait'
coef_cca


# %%
# General study
# =============

# dimensions of the simulated data
m = 13000 # sample size
n_traits = 8 # number of dimensions
trait_names = [f'X{i}' for i in range(1,n_traits+1)]

# Number of subjects (different sub-population) in each trait
dic_m = {"m1":4000, 
         "m2":500, 
         "m3":500, 
         "m4":50, 
         "m5":50, 
         "m6":2000, 
         "m7":2900, 
         "m8":3000} # m8 = m -m1 -m2 -m3 -m4 -m5 -m6 -m7


# %%
# Generate the n_traits sparse phenotypes
#

study_X = generate_traits( np.array(list(dic_m.values())) )

study_corr = np.corrcoef(study_X, rowvar=False)


# %%
# Generate the associated genotypes (one per sub-population)
#

study_genotype = np.concatenate([np.linspace(0, 2, dic_m["m1"]), np.linspace(0, 2, dic_m["m2"]), 
                           np.linspace(0, 2, dic_m["m3"]), np.linspace(0, 2, dic_m["m4"]), 
                           np.linspace(0, 2, dic_m["m5"]), np.linspace(0, 2, dic_m["m6"]),
                           np.linspace(0, 2, dic_m["m7"]), np.linspace(0, 2, dic_m["m8"])])
study_genotype = study_genotype.round()

list_noise = np.linspace(4,5,2)
list_most_pval = []
list_multi_pval = []
dic_univ_betas = {}
dic_multiv_betas = {}
dic_cca_betas = {}
dic_fake_h2 = {}

for noise_level in list_noise:
    fake_h2 = []
    for i in range(n_traits):
        rand_noise = 2*np.random.rand()
        study_X[:,i] += np.random.randn(m)*(noise_level + rand_noise)
        fake_h2.append(dic_m[f"m{i+1}"]/np.array(list(dic_m.values())).sum() * (noise_level + rand_noise)**2)

    dic_fake_h2[noise_level] = fake_h2

    mpheno_geno = pd.DataFrame(study_X, columns=[f'X{i}' for i in range(1,n_traits+1)])
    mpheno_geno['Genotype'] = study_genotype
    list_mahalanobis_norm_perm = mahalanobis_norm_perm(mpheno_geno, study_corr, nb_perm_geno = 5000)

    univariates_gwases, list_betas_orig, list_z_score_orig = UniVar_reg(mpheno_geno, trait_names)
    z_score_observed = univariates_gwases['t-value'].values
    dic_univ_betas[noise_level] = np.array(z_score_observed)/z_score_observed[0]

    alpha, loc, scale = fit_gamma(list_mahalanobis_norm_perm, plot_fit=False)
    most_pval = MOSTest(z_score_observed, study_corr, alpha, loc, scale)
    list_most_pval.append(most_pval)

    pheno = sm.add_constant(mpheno_geno[[f'X{i}' for i in range(1,n_traits+1)]])
    model = sm.OLS(mpheno_geno['Genotype'], pheno)
    results = model.fit()
    multi_pval = results.f_pvalue

    dic_multiv_betas[noise_level] = np.array(results.params[[f'X{i}' for i in range(1,n_traits+1)]].to_list())/results.params['X1']
    list_multi_pval.append(multi_pval)

    cca = CCA(n_components=1)
    cca.fit(pheno, mpheno_geno['Genotype'])
    dic_cca_betas[noise_level] = np.array(cca.coef_[0, 1:])/cca.coef_[0, 1]


# %%
# Results plot vs Signal to Noise ratio
# -------------------------------------
#
# Evolution of the estimated betas (comparison between univariate and
# multivariate approaches)

plt.figure(figsize=(18,18))
for i in range(n_traits):
    plt.subplot(n_traits//2, 2, i+1)
    plt.plot(list_noise, [dic_cca_betas[noise][i] for noise in list_noise], label=f'CCA beta{i}')
    plt.plot(list_noise, [dic_multiv_betas[noise][i] for noise in list_noise], label=f'MultiVariate beta{i}')
    plt.plot(list_noise, [dic_univ_betas[noise][i] for noise in list_noise], label=f'Univ beta{i}')
    plt.legend()
    plt.title('Proportion of the related sub-population: '+ format(dic_m[f"m{i+1}"]/np.array(list(dic_m.values())).sum(), '.3f'))


# %%
# Evolution of the -log10(p-value) as a function of the signal to noise ratio

plt.figure(figsize=(12,6))
plt.plot(list_noise, -np.log10(np.array(list_multi_pval)), label='MultiVariate')
plt.plot(list_noise, -np.log10(np.array(list_most_pval)), label='MOSTest')
plt.title('Evolution of the -log10(p-value) as a function of the signal to noise volume')
plt.legend()
plt.show()


print(format(list_multi_pval[-1], '.2'))
print(format(list_most_pval[-1], '.2'))
