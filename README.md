# star_params
stellar parameter estimation 


## Idea
- How accurate/precise can MIST stellar evolutionary model in isochrones grids in deriving stellar parameters?
- Stars in the main sequence (MS) phase evolve slowly over millions of years so their stellar parameters are often degenerate with MS age 
- Stars can have less degenerate stellar parameters when they are evolving quickly during their younger phases e.g. ZAMS<EEP<TAMS

## Methods
1. Compile a list of known young (t<100Myr) transiting host stars with precise published ages 
 - [young_transiting_planets.ipynb](./notebooks/young_transiting_planets.ipynb)
2. Using [isochrones](https://github.com/timothydmorton/isochrones) grid interpolation, determine which observables (e.g. photometric measurements) are most important to derive fundamental stellar parameters (e.g. radius, mass, T_eff, [Fe]/[H], log(g))
 - [experiments](./isochrones_runs)
3. How accurate can these stars be determined using the same techniques?