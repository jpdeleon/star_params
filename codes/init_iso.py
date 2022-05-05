#import sys
#sys.path.append('../codes')
from logging import exception
import pdb
import numpy as np
from tqdm import tqdm

from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd
from isochrones.priors import DistancePrior, GaussianPrior, FlatPrior

from star import Star

use_priors = True
refit = True
verbose = True
redownload = False
fp_data = '../data/nexsci_data.csv'
bands_to_use = "J H K BP RP".split()
nexsci_pars = "teff logg met dist".split()
cols_corner = 'radius mass Teff logg feh age'.split()

if redownload:
    #pstable combines data from the Confirmed Planets and Extended Planet Parameters tables
    tab = NasaExoplanetArchive.query_criteria(table="pscomppars", 
                                              where="discoverymethod like 'Transit'"
                                            )
    df = tab.to_pandas()
    df.to_csv(fp_data, index=True)
else:
    df = pd.read_csv(fp_data, index_col=0)
    df.index.name = 'index'
# df.head()
idx = (df.st_age<0.1) & (df.st_ageerr1<0.5)
df_young = df[idx]
# df_young
host_names = list(set(df_young['hostname']))
#host_names


errors = {}
for name in tqdm(host_names):
    d = df_young.query(f"pl_name=='{name} b'")    

    props = {}
    for p in nexsci_pars:
        if p=='met':
            props['feh'] = (d[[f'st_{p}',f'st_{p}err1']].values[0])
        elif p=='teff':
            props['Teff'] = d[[f'st_{p}',f'st_{p}err1']].values[0]
        elif p=='dist':
            props[p] = d[[f'sy_{p}',f'sy_{p}err1']].values[0]
        else:
            props[p] = d[[f'st_{p}',f'st_{p}err1']].values[0]
    props = {p: tuple(props[p]) for p in props}

    if verbose:
        print("Initiating star.")
    try:
        if name=='DS Tuc A':
            name='DS Tuc'
        s = Star(name=name, verbose=verbose, search_radius=30)
        iso_params = s.get_iso_params(teff=props['Teff'],
                    logg=props['logg'],
                    feh=props['feh'],
                    bands=bands_to_use
                    )
        iso_model = s.init_isochrones(iso_params=iso_params)
        # import pdb; pdb.set_trace()
        if use_priors:
            iso_model.set_prior(
                    # feh=GaussianPrior(*props['feh']),
                    # distance=GaussianPrior(*props['dist']), #bounded prior with mean and sigma 
                    distance=DistancePrior(2*props['dist'][0]),
                    age=FlatPrior((6,9)), #a flat log prior
            )
        # if verbose:
        #     print(f"Running isochrones using:\n{iso_params}\n.")
        iso_model = s.run_isochrones(iso_params=iso_params, 
                        overwrite=refit,
                        n_live_points=1000 #multinest parameter
                        )
        fp = '../isochrones_runs/'+name.replace(' ', '_')

        fig1 = s.plot_corner(posterior="observed")
        if use_priors:
            fp1 = fp+'/corner_obs_with_priors.png'
        else:
            fp1 = fp+'/corner_obs.png'
        fig1.savefig(fp1, bbox_inches=False)
        
        truths = (
            d['st_rad'].values[0],
            d['st_mass'].values[0],
            props['Teff'][0],
            props['logg'][0],
            props['feh'][0],
            np.log10(d['st_age'].values[0]*1e9),
            )

        fig2 = s.isochrones_model.corner_derived(cols_corner, 
                                            quantiles=(0.16,0.84), 
                                            show_titles=True,
                                            truths=truths
                                            )
        if use_priors:
            fp2 = fp+'/corner_phys_with_priors.png'
        else:
            fp2 = fp+'/corner_phys.png'
        fig2.savefig(fp2, bbox_inches=False)
    except Exception as e:
        errors[name] = e
        print("Error: ", e)

if len(errors)>0:
    print(errors)