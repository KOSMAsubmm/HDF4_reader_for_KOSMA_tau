# HDF4_reader_for_KOSMA_tau

This routine is used to analyse the hdf files of the kosma- tau PDR model.
Kosma-tau PDR model is used to understand the chemistry and physics of the molecular cloud.
### Pre-requisites:
    - `numpy`
    - `pandas`
    - `matplotlib`
    - `datetime`
    - `os`
    - `pyhdf`
    - `h5py`

## The main functions are:
## `read_hdf(f)`
Read the hdf file 'f' and return a pandas dataframe (df), heating and cooling rates (HC_rates), and level population details (rad_field). After reading the hdf4 file into a dataframe one can use the following functions to plot different properties

## `plot_temperature_profile(df)` 
to plot the gas and dust temperature, returns the figure

## `plot_abundance_profile(df, sp)`
plot the abundance profile of the species sp, returns the figure

## `plot_column_density_profile(df, sp) `
 plot the column density profile of the species sp, returns the figure
 

## `plot_diffusion_properties(df, key ,sp)`
plot the diffusion properties of the species sp. different key available are:

### - K : plot thermal, molecular and turbulent diffusion coefficients of the species
### - V : plot thermal, molecular and turbulent diffusion velocities of the species
### - dr : plot total, thermal, molecular and turbulent diffusion rates of the species. Both formation and destruction rates are indicated using different markers.
