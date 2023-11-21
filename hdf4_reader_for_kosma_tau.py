# This routine contains every program to analyse the results from the 
# kosma-tau PDR model.
# created by Aleena Baby as part of the Ph.D. thesis

# ###############################################################
#   importing all the neccessary modules
# ###############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from matplotlib.backends.backend_pdf import PdfPages
import pyhdf.SD as SD
from pyhdf.HDF import*
from pyhdf.VS import*
import h5py
import warnings
import matplotlib.font_manager as font_manager
warnings.filterwarnings("ignore")
from matplotlib import rc
rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})
rc('text', usetex = True)

SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 40

plt.rc('font', size = SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize = SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize = BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize = SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['axes.linewidth'] = 3


# ###############################################################
#   functions to read the hdf file
# ###############################################################

ctrl_ind = '~/HDF4_reader_for_KOSMA_tau/CTRL_IND'

pc_to_cm = 3.0856e18

def read_cntrl_ind_file(f):
    file = pd.read_fwf(f, header = None)
    index = file[file[0].str.contains('# DENSITY  ')].index.values[:-1]
    molecule = []
    spec_ind = []
    for ind in index:
        line = file[file.index == ind].values
        foo, quantity, species, number = line[0][0].split()
        molecule.append(species)
        spec_ind.append(int(number))
    if len(molecule) == len(spec_ind):
        print ('there are ',len(molecule),'number of species in this model')
    return (molecule, spec_ind)


molecule, spec_ind = read_cntrl_ind_file(ctrl_ind)

list_of_species_and_numbers = {A:B for A,B in zip(molecule, spec_ind)}

sp_name = { 'ELECTR' :'e-', 'EL' :'e-','Elect' :'e$^-$', 'H':'H', 'H2': 'H$_2$',\
           'H_2': 'H$_2$', 'H+': 'H$^+$', 'H2+': 'H$_2^+$','H_2+': 'H$_2^+$' , \
           'H2*': 'H$_2*$', 'H3+': 'H$_3^+$', 'HE+': 'He$^+$','He+': 'He$^+$','H_3+': 'H$_3^+$', 'HE': 'He', 'He': 'He', \
           'O+': 'O$^+$', 'O':'O', 'C+':'C$^+$', '13C+': '$^{13}$C$^+$' , 'OH+': 'OH$^+$'\
           , 'O2': 'O$_2$' , 'SO2': 'SO$_2$', 'CO+': 'CO$^+$', 'SO+': 'SO$^+$', 'CH+': 'CH$^+$', \
           'CH2': 'CH$_2$', 'HS+': 'HS$^+$', '13C': '$^{13}$C', '13CO': '$^{13}$CO', \
           '13CO+': '$^{13}$CO$^+$', '13C18O': '$^{13}$C$^{18}$O', '13C18O+': '$^{13}$C$^{18}$O$^+$', \
           'C18O': 'C$^{18}$O', '13CH': '$^{13}$CH' , '13CH+': '$^{13}$CH$^+$', 'H2O': 'H$_2$O', \
           'H2O+': 'H$_2$O$^+$', 'H13CO+': 'H$^{13}$CO$^+$', 'HCS+': 'HCS$^+$', 'CS+': 'CS$^+$', \
           'CH2+': 'CH$_2^+$', '13CH2+': '$^{13}$CH$_2^+$', '13CH2': '$^{13}$CH$_2$', 'H3O+': 'H$_3$O$^+$', 'H3': 'H$_3$', \
           'HCO+': 'HCO$^+$', 'S+': 'S$^+$' , 'CH3+': 'CH$_3^+$', 'O18O': 'O$^{18}$O', \
           '18O': '$^{18}$O' , '18OH': '$^{18}$OH', '18O+': '$^{18}$O$^+$', '18OH+': '$^{18}$OH$^+$', \
           'H13C18O+':'H$^{13}$C$^{18}$O$^+$', 'HC18O+':'HC$^{18}$O$^+$', 'H218O+':'H$_2^{18}$O$^+$', \
           'C18O+': 'C$^{18}$O$^+$', 'C':'C', 'OH':'OH', 'CO':'CO', 'CH':'CH', 'SO':'SO' \
           , 'S':'S', 'OCS':'OCS', 'HS':'HS', 'H2S+':'H$_2$S$^+$', 'CS':'CS', 'OCS+':'OCS$^+$' \
           , 'H318O+':'H$_3^{18}$O$^+$', 'H218O':'H$_2^{18}$O', 'PHOTON': '$\gamma_{FUV}$'  \
           , 'CRPHOT': 'CR$_{Phot}$' , '==>': 'diffusion','diffusion': 'diffusion', '-->': '-->'}

hcreat_name = {'H2deexcitation':'H$_2$ deexcitation', 'H2photodissheating':'H$_2$ photo-diss heating', 'H2formation':'H$_2$ formation', \
          'OI_63':'OI(63 $\mu $m)', 'OI_44':'OI(44 $\mu $m)', 'OI_146':'OI(146 $\mu$ m)', 'cosmicray':'cosmic ray', \
          'PE':'PE', '12CO':'$^{12}$CO', 'CII158':'[CII] (158 $\mu$ m)', 'CI610':'[CI] (610 $\mu$ m)', \
          'CI230':'[CI] (230 $\mu$ m)', 'CI370':'[CI] (370 $\mu$ m)', 'SiII35':'[SiII](35 $\mu$ m)', \
          '13COcooling':'$^{13}$CO cooling', 'Lymanalpha':'Lyman- alpha', 'H2O':'H$_2$O', 'gas_grain':'gas-grain', 'OH':'OH', 'OI6300':'OI (6300 $\mu$ m)', \
          'H2photodisscooling':'H$_2$ photo-diss cooling', 'Cioniationheating':'C-ioniation heating', 'chemicalheating':'chemical heating'}

h2_hc_keys = ['H2deexcitation', 'H2photodissheating', 'H2formation','Lymanalpha', 'H2O','gas_grain', 'OH','H2photodisscooling']
oi_hc_keys = ['OI_63','OI_44', 'OI_146','OH', 'OI6300']
c_hc_keys = ['12CO', 'CII158', 'CI610', 'CI230', 'CI370','13COcooling','Cioniationheating']
res_hc_keys = ['cosmicray', 'PE','chemicalheating']
heating_cooling_keys = [h2_hc_keys, oi_hc_keys, c_hc_keys, res_hc_keys]
keys_rad_field = ['H2dissrate', 'Cionisation', 'COdissrate','CO12selfshield', 'CO13selfshield', 'H2formrate', 'H2rovibrate']

rad_field_name = {'H2dissrate':'H$_2$ dissociation rate', 'Cionisation':'C ionisation', 'COdissrate':'CO dissociation rate', \
                  'CO12selfshield':'$^{12}$CO selfshielding', 'CO13selfshield':'$^{13}$CO selfshielding',\
                  'H2formrate':'H$_2$ formation rate','H2rovibrate': 'H$_2$ ro-vib rate'}

debug = False
unitofk = '[$\\rm{cm^{2}\\,s^{-1}}$]'
unit_vkms = 'V [$\\rm{km\\,s^{-1}}$]'

K17 = '$ \\rm K_{turb}$ = $10^{17}$' + unitofk
K15,K16, K18 = '$ \\rm K_{turb}$ = $10^{15}$ '+unitofk,'$ \\rm K_{turb}$ = $10^{16}$ '+unitofk,'$ \\rm K_{turb}$ = $10^{18}$ '+unitofk
K19, K20, K21, K22 = '$ \\rm K_{turb}$ = $10^{19}$ '+unitofk, '$ \\rm K_{turb}$ = $10^{20}$ '+unitofk, '$ \\rm K_{turb}$ = $10^{21}$ '+unitofk , '$ \\rm K_{turb}$ = $10^{22}$ '+unitofk
K23 = '$ \\rm K_{turb}$ = $10^{23}$ '+unitofk
K0 = 'K = 0 '+unitofk
rates = 'rates [$\\rm{cm^{-3}s^{-1}}$]'


def read_hdf(f):
    if not h5py.is_hdf5(f): #to check whether hdf5 or hdf4 format
        print ('reading hdf4 file', f)
        hdf = SD.SD(f)
        dataset = hdf.datasets()
        print (dataset)
        sds = hdf.select('Depth::Temp::Abundances').get()
        #Av is 0 and pc is 1
        depths = sds[1, :]
        av = sds[0, :]
        tg = sds[2, :]
        td = sds[3, :]
        print ("maximum temperature = ", max(tg))
        abundances = sds[804:866, :]
        sds = hdf.select('Column-densities').get()
        CDD = sds[804:866, :] 
        HC_rates = hdf.select('Heating-Cooling-Rates').get()
        optical_depth = hdf.select('Optical-depths').get()
        rad_field = hdf.select('Rad-field::photorates').get()
        d = 'diffusion rates'
        if d in dataset:
            diff_rates = hdf.select('diffusion rates').get()
        else:
            diff_rates = np.zeros(shape = (400, len(depths)))
        rows = []
        for molecule_name, abund, depth in zip(molecule, abundances, depths):
            for it, ab in enumerate(abund):
                mol = molecule.index(molecule_name)
                rows.append({"it":it, "gas_temp": tg[it],  "dust_temp": td[it], "molecule":molecule_name, "depth":depths[it], 'Av':av[it], "abund":ab, \
                    "cdd":CDD[mol, it], "total_diff_rates": diff_rates[mol, it], "mol_diff_rates": diff_rates[mol+65, it], \
                            "therm_diff_rates": diff_rates[mol+130, it], "turb_diff_rates": diff_rates[mol+195, it], "K_turb": diff_rates[mol+260, it], \
                           "A1": diff_rates[mol+325, it], "A2": diff_rates[mol+390, it], "K_th": diff_rates[mol+455, it]})
                
        df = pd.DataFrame(rows)        
    else:
        print ('It is a hdf5 file. Please use arrays_h5 function to read') 
    return (df, HC_rates, rad_field)


def make_figure(one, SMALL_SIZE):
    fig = plt.figure(figsize = (one,one/1.33))#(9,6.9))
    ax = fig.add_subplot(111)
    #SMALL_SIZE = 20
    MEDIUM_SIZE = SMALL_SIZE + 4
    BIGGER_SIZE = SMALL_SIZE +6
    #plt.rcParams['figure.figsize'] = (7,5.25)
    plt.rcParams['figure.dpi'] = (150)
    plt.rc('font', size = SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize = SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize = MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize = SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams['axes.linewidth'] = 1.3
    plt.rc('lines', linewidth= 4)
    ax. xaxis. set_tick_params (which = 'major', size = 5, width = 1, \
                        direction = 'in', top = 'on', pad = 10)
    ax. xaxis. set_tick_params (which = 'minor', size = 2, width = 1, \
                        direction = 'in', top = 'on', pad = 10)
    ax. yaxis. set_tick_params (which = 'major', size = 5, width = 1, \
                        direction = 'in', top = 'on', pad = 10)
    ax. yaxis. set_tick_params (which = 'minor', size = 2, width = 1, \
                        direction = 'in', top = 'on', pad = 10)
    xtick_labels = ax.get_xticklabels()
    # Set the font properties of the xtick labels to bold
    bold_font = font_manager.FontProperties(weight='bold')
    for label in xtick_labels:
        label.set_fontproperties(bold_font)
    return (fig, ax)


def plot_abundance_profile(df, sp):
    fig, ax = make_figure(5,13)
    ax.set_xlabel('r [$\\rm{pc}$]', fontsize = 18)
    xlim = [1.0e-4,df.depth.unique[-1]]
    ax.set_xlim(xlim)
    ax.set_ylabel('$n_i/n$', fontsize = 18)
    ax.semilogx(df[df.molecule == sp].depth,\
                df[df.molecule == sp]['abundance'],'-',\
                    linewidth = 2.5, label = sp_name)
    ax.legend(loc = 'lower left', frameon = False, fontsize = 14)
    fig.tight_layout()
    return (fig)

def plot_column_density_profile(df, sp):
    fig, ax = make_figure(5,13)
    ax.set_xlabel('r [$\\rm{pc}$]', fontsize = 18)
    xlim = [1.0e-4,df.depth.unique[-1]]
    ax.set_xlim(xlim)
    ax.set_ylabel('$N_i$', fontsize = 18)
    ax.semilogx(df[df.molecule == sp].depth,\
                df[df.molecule == sp]['cdd'],'-',\
                    linewidth = 2.5, label = sp_name)
    ax.legend(loc = 'lower left', frameon = False, fontsize = 14)
    fig.tight_layout()
    return (fig) 
    
def plot_temperature_profile(df):
    fig, ax = make_figure(5,13)
    ax.set_xlabel('r [$\\rm{pc}$]', fontsize = 18)
    xlim = [1.0e-4,df.depth.unique[-1]]
    ax.set_xlim(xlim)
    ax.set_ylabel('T [$\\rm{K}$]', fontsize = 18)
    sp = 'H'
    ax.semilogx(df[df.molecule == sp].depth,\
                df[df.molecule == sp]['gas_temp'],'-',\
                    linewidth = 2.5, label = '$\\rm{T_{gas}}$')
    ax.semilogx(df[df.molecule == sp].depth,\
                df[df.molecule == sp]['dust_temp'],'-',\
                    linewidth = 2.5, label = '$\\rm{T_{dust}}$')
    ax.legend(loc = 'lower left', frameon = False, fontsize = 14)
    fig.tight_layout()
    return (fig)

def plot_diffusion_properties(df, key ,sp):
    fig, ax = make_figure(5,13)
    ax.set_xlabel('r [$\\rm{pc}$]', fontsize = 18)
    xlim = [1.0e-4,df.depth.unique[-1]]
    ax.set_xlim(xlim)
    if key == 'K':
        ax.set_ylabel('K [$\\rm{cm^2s^{-1}}$]', fontsize = 18)
        key1, key2, key3 = Ktherm, Kmol, Kturb
    if key == 'V':
        ax.set_ylabel('V [$\\rm{cms^{-1}}$]', fontsize = 18)
        key1, key2, key3 = V_therm, V_mol, V_turb
    ax.semilogx(df[df.molecule == sp].depth,\
                df[df.molecule == sp][key1],'-',\
                    linewidth = 2.5, label = 'thermal')
    ax.semilogx(df[df.molecule == sp].depth,\
                df[df.molecule == sp][key2],'-',\
                    linewidth = 2.5, label = 'molecular')
    ax.semilogx(df[df.molecule == sp].depth,\
                df[df.molecule == sp][key3],'-',\
                    linewidth = 2.5, label = 'turbulent')
    ax.legend(loc = 'lower left', frameon = False, fontsize = 14)
    if key == 'dr':
        linet, = ax.loglog(df[df.molecule == sp][xkey], abs(df[df.molecule == sp].total_diff_rates), \
                    '-',color = 'g', alpha =0.3, linewidth = 2, label = 'total')
        ykey = 'turb_diff_rates'
        linetu, = ax.loglog(df[df.molecule == sp][xkey], df[df.molecule == sp][ykey], \
                    '*',markerfacecolor = 'None',markeredgecolor = default_blue, markersize = 6, label = 'turbulent')
        k = df[df.molecule == sp]
        l = k[k[ykey] <= 0.0]
        line1, = ax.loglog(l[xkey], abs(l[ykey]),'*',markerfacecolor = 'None',markeredgecolor = default_orange, markersize = 6)

        ykey = 'therm_diff_rates'
        lineth, = ax.loglog(df[df.molecule == sp][xkey], df[df.molecule == sp][ykey], \
                    'o',markerfacecolor = 'None',markeredgecolor = default_blue, markersize = 4, label = 'thermal')
        k = df[df.molecule == sp]
        l = k[k[ykey] <= 0.0]
        line1, = ax.loglog(l[xkey], abs(l[ykey]),'o',markerfacecolor = 'None',markeredgecolor = default_orange, markersize = 4)

        ykey = 'mol_diff_rates'
        lineml, = ax.loglog(df[df.molecule == sp][xkey], df[df.molecule == sp][ykey], \
                    's',markerfacecolor = 'None',markeredgecolor = default_blue, markersize = 4, label = 'molecular')
        k = df[df.molecule == sp]
        l = k[k[ykey] <= 0.0]
        line1, = ax.loglog(l[xkey], abs(l[ykey]),'s',markerfacecolor = 'None',markeredgecolor = default_orange, markersize = 4)

        leg1 = ax.legend([line1,lineml, line1],[total,'formation','destruction'],loc = 'upper right', \
              ncol = 3,fontsize = 15,frameon = False, handletextpad = 0.02,\
                              columnspacing = 0.11)
        ax.legend(loc = (0.4,0.6), fontsize = 16, ncol = 2,frameon = False, handletextpad = 0.6,\
                                  columnspacing = 0.8, handlelength = 0.8)#
        ax.add_artist(leg1)
    fig.tight_layout()
    return (fig)


def plot_temperature_profile(df):
    for hc_key in heating_cooling_keys:
        fig, ax = make_figure(5,13)
        ax.set_xlabel('r [$\\rm{pc}$]', fontsize = 18)
        xlim = [1.0e-4,df.depth.unique[-1]]
        ax.set_xlim(xlim)
        ax.set_ylabel('Heating/cooling rates [$\\rm{ergcm^{-3}s^{-1}}$]', fontsize = 15)
        for hc in hc_key:
            ax.semilogx(df.depth, df[hc],'-',\
                        linewidth = 2.5, label = hcreat_name[hc])
        ax.legend(loc = 'lower left', frameon = False, fontsize = 14)
        fig.tight_layout()
    return (fig)