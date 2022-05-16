# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

from utils import get_mag_err_from_flux, get_mist_eep_table, map_float
from target import Target

class Star(Target):
    """
    Performs physics-related calculations for stellar characterization.
    Inherits the Target class.
    """

    def __init__(
        self,
        name=None,
        toiid=None,
        ticid=None,
        epicid=None,
        gaiaDR2id=None,
        ra_deg=None,
        dec_deg=None,
        mission="tess",
        search_radius=3,
        prot=None,
        mcmc_steps=1000,
        burnin=500,
        thin=1,
        alpha=(0.56, 1.05),  # Morris+2020
        slope=(-0.50, 0.17),  # Morris+2020
        sigma_blur=3,
        use_skew_slope=False,
        nsamples=1e4,
        verbose=True,
        clobber=True,
    ):
        """
        Attributes
        ----------
        See inherited class: Target

        See starfit:
        https://github.com/timothydmorton/isochrones/blob/master/isochrones/starfit.py
        """
        # https://docs.python.org/3/library/inspect.html#inspect.getdoc
        super().__init__(
            name=name,
            toiid=toiid,
            ticid=ticid,
            epicid=epicid,
            gaiaDR2id=gaiaDR2id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            search_radius=search_radius,
            verbose=verbose,
            clobber=clobber,
            mission=mission,
        )
        self.mcmc_steps = mcmc_steps
        self.burnin = burnin
        self.thin = thin
        self.prot = prot
        self.alpha = alpha
        self.slope = slope
        self.sigma_blur = sigma_blur
        self.use_skew_slope = use_skew_slope
        self.nsamples = int(nsamples)
        self.isochrones_model = None
        self.stardate = None
        self.iso_params = None
        self.iso_param_names = [
            "EEP",
            "log10(Age [yr])",
            "[Fe/H]",
            "ln(Distance)",
            "Av",
        ]
        self.iso_params0 = (329.58, 9.5596, -0.0478, 5.560681631015528, 0.0045)
        self.iso_params_init = {
            k: self.iso_params0[i] for i, k in enumerate(self.iso_param_names)
        }
        self.perc = [16, 50, 84]
        vizier = self.query_vizier(verbose=False)
        self.starhorse = (
            vizier["I/349/starhorse"]
            if "I/349/starhorse" in vizier.keys()
            else None
        )
        self.mist = None
        self.mist_eep_table = get_mist_eep_table()
        
    def estimate_Av(self, map="sfd", constant=None):
        """
        compute the extinction Av from color index E(B-V)
        estimated from dustmaps via Av=constant*E(B-V)

        Parameters
        ----------
        map : str
            dust map
        See below for conversion from E(B-V) to Av:
        https://dustmaps.readthedocs.io/en/latest/examples.html
        """
        try:
            import dustmaps
        except Exception:
            raise ModuleNotFoundError("pip install dustmaps")

        if map == "sfd":
            from dustmaps import sfd

            # sfd.fetch()
            dust_map = sfd.SFDQuery()
            constant = 2.742 if constant is None else constant
        elif map == "planck":
            from dustmaps import planck

            # planck.fetch()
            dust_map = planck.PlanckQuery()
            constant = 3.1 if constant is None else constant
        elif map == "bayestar":
            from dustmaps import bayestar

            bayestar.BayestarQuery()
            dust_map = 2.742 if constant is None else constant
        else:
            raise ValueError("Available maps: (sfd,planck,bayestar)")

        ebv = dust_map(self.target_coord)
        Av = constant * ebv
        return Av
    
    def get_iso_params(
        self,
        teff=None,
        logg=None,
        feh=None,
        add_parallax=True,
        add_dict=None,
        bands=["J", "H", "K"],  # "G BP RP J H K W1 W2 W3 TESS".split(),
        correct_Gmag=True,
        plx_offset=-0.08,
        inflate_plx_err=True,
        min_mag_err=0.01,
    ):
        """get parameters for isochrones

        Parameters
        ----------
        teff, logg, feh: tuple
            'gaia' populates Teff from gaia DR2
        bands : list
            list of photometric bands
        add_parallax : bool
            default=True
        add_dict : dict
            additional params
        correct_Gmag : bool
            inflate Gmag and Gmag err (Casagrande & VandenBerg 2018)
        plx_offset : float
            systematic parallax offset (default=-80 uas, Stassun & Torres 2018)
        inflate_plx_err : bool
            adds 0.01 parallax error in quadrature (default=True) (Luri+2018)
        min_mag_err : float
            minimum magnitude uncertainty to use
        Returns
        -------
        iso_params : dict
        """
        errmsg = "`bands must be a list`"
        assert isinstance(bands, list), errmsg
        if self.gaia_params is None:
            gp = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)
        else:
            gp = self.gaia_params
        if self.tic_params is None:
            tp = self.query_tic_catalog(return_nearest_xmatch=True)
        else:
            tp = self.tic_params
        if not self.validate_gaia_tic_xmatch():
            msg = f"TIC {self.ticid} does not match Gaia DR2 {self.gaiaid} properties"
            raise Exception(msg)

        params = {}
        # spectroscopic constraints
        if teff == "gaia":
            # Use Teff from Gaia by default
            teff = gp["teff_val"]
            teff_err = np.hypot(
                gp["teff_percentile_lower"], gp["teff_percentile_lower"]
            )
            if not np.any(np.isnan(map_float((tp["Teff"], tp["e_Teff"])))):
                if teff_err > tp["e_Teff"]:
                    # use Teff from TIC if Teff error is smaller
                    teff = tp["Teff"]
                    teff_err = tp["e_Teff"]
            params.update({"teff": (teff, teff_err)})
        elif teff is not None:
            assert isinstance(
                teff, tuple
            ), "teff must be a tuple (value,error)"
            teff, teff_err = teff[0], teff[1]
            params.update({"Teff": (teff, teff_err)})
        if feh is not None:
            # params.update({"feh": (tp["MH"], tp["e_MH"])})
            assert isinstance(feh, tuple), "feh must be a tuple (value,error)"
            params.update({"feh": (feh[0], feh[1])})
        if logg is not None:
            # params.update({"logg": (tp["logg"], tp["e_logg"])})
            assert isinstance(
                logg, tuple
            ), "logg must be a tuple (value,error)"
            params.update({"logg": (logg[0], logg[1])})
        if add_parallax:
            plx = gp["parallax"] + plx_offset
            if inflate_plx_err:
                # inflate error based on Luri+2018
                plx_err = np.hypot(gp["parallax_error"], 0.1)
            else:
                plx_err = gp["parallax_error"]
            params.update({"parallax": (plx, plx_err)})

        # get magnitudes from vizier
        mags = self.query_vizier_mags()
        if (self.ticid is not None) and (self.toi_params is not None):
            Tmag_err = (
                self.toi_Tmag_err
                if self.toi_Tmag_err > min_mag_err
                else min_mag_err
            )
            if "TESS" in bands:
                params.update({"TESS": (self.toi_Tmag, Tmag_err)})
        if "G" in bands:
            if correct_Gmag:
                # Casagrande & VandenBerg 2018
                gmag = gp["phot_g_mean_mag"] * 0.9966 + 0.0505
            else:
                gmag = gp["phot_g_mean_mag"]
            gmag_err = get_mag_err_from_flux(
                gp["phot_g_mean_flux"], gp["phot_g_mean_flux_error"]
            )
            gmag_err = gmag_err if gmag_err > min_mag_err else min_mag_err
            params.update({"G": (gmag, gmag_err)})
        if "BP" in bands:
            bpmag = gp["phot_bp_mean_mag"]
            bpmag_err = get_mag_err_from_flux(
                gp["phot_bp_mean_flux"], gp["phot_bp_mean_flux_error"]
            )
            bpmag_err = bpmag_err if bpmag_err > min_mag_err else min_mag_err
            params.update({"BP": (bpmag, bpmag_err)})
        if "RP" in bands:
            rpmag = gp["phot_rp_mean_mag"]
            rpmag_err = get_mag_err_from_flux(
                gp["phot_rp_mean_flux"], gp["phot_rp_mean_flux_error"]
            )
            rpmag_err = rpmag_err if rpmag_err > min_mag_err else min_mag_err
            params.update({"RP": (rpmag, rpmag_err)})
        if "B" in bands:
            # from tic catalog
            params.update({"B": (tp["Bmag"], tp["e_Bmag"])})
        if "V" in bands:
            # from tic catalog
            params.update({"V": (tp["Vmag"], tp["e_Vmag"])})
        if "J" in bands:
            # from tic catalog
            params.update({"J": (tp["Jmag"], tp["e_Jmag"])})
        if "H" in bands:
            # from tic catalog
            params.update({"H": (tp["Hmag"], tp["e_Hmag"])})
        if "K" in bands:
            # from tic catalog
            params.update({"K": (tp["Kmag"], tp["e_Kmag"])})
        # WISE
        for b in bands:
            if b[0] == "W":
                if f"{b}mag" in mags.index.tolist():
                    wmag = mags[f"{b}mag"]
                    wmag_err = mags[f"e_{b}mag"]
                    wmag_err = (
                        wmag_err if wmag_err > min_mag_err else min_mag_err
                    )
                    params.update({b: (round(wmag, 2), round(wmag_err, 2))})
                else:
                    print(f"{b} not in {mags.index.tolist()}")
            elif b[:2] == "Kp":
                if f"{b}mag" in mags.index.tolist():
                    wmag = mags[f"{b}mag"]
                    wmag_err = mags[f"e_{b}mag"]
                    wmag_err = (
                        wmag_err if wmag_err > min_mag_err else min_mag_err
                    )
                    params.update({b: (round(wmag, 2), round(wmag_err, 2))})
                else:
                    print(f"{b} not in {mags.index.tolist()}")

        # adds and/or overwrites above
        if add_dict is not None:
            assert isinstance(add_dict, dict)
            params.update(add_dict)
        # remove nan if there is any
        iso_params = {}
        for k in params:
            vals = map_float(params[k])
            if np.any(np.isnan(vals)):
                print(f"{k} is ignored due to nan ({vals})")
            else:
                iso_params[k] = vals
        self.iso_params = iso_params
        return iso_params
    
    def save_ini_isochrones(self, outdir=".", header=None, **iso_kwargs):
        """star.ini file for isochrones starfit script
        See:
        https://github.com/timothydmorton/isochrones/blob/master/README.rst
        """
        target_name = self.target_name.replace(" ", "")
        if self.iso_params is None:
            iso_params = self.get_iso_params(**iso_kwargs)
        else:
            iso_params = self.iso_params
        starfit_arr = []
        for k in iso_params:
            vals = map_float(iso_params[k])
            if np.any(np.isnan(vals)):
                print(f"{k} is ignored due to nan ({vals})")
            else:
                if k in ["Teff", "parallax", "logg", "feh"]:
                    par = k
                else:
                    # photometry e.g. J, H, K
                    par = k.upper()
                if k == "Teff":
                    starfit_arr.append(f"{par} = {vals[0]:.0f}, {vals[1]:.0f}")
                else:
                    starfit_arr.append(f"{par} = {vals[0]:.3f}, {vals[1]:.3f}")
        if self.mission.lower() == "k2":
            q = self.query_vizier_param("Kpmag")
            if "IV/34/epic" in q:
                Kpmag = q["IV/34/epic"]
                starfit_arr.append(f"Kepler = {Kpmag:.3f}")

        outdir = target_name if outdir == "." else outdir
        outpath = Path(outdir, "star.ini")
        if not Path(outdir).exists():
            Path(outdir).mkdir()
        header = target_name if header is None else header
        np.savetxt(outpath, starfit_arr, fmt="%2s", header=header)
        print(f"Saved: {outpath}\n{starfit_arr}")
        
    def init_isochrones(
        self,
        iso_params=None,
        model="mist",
        maxAV=None,
        max_distance=None,
        bands=None,
        binary_star=False,
    ):
        """initialize parameters for isochrones

        Parameters
        ----------
        iso_params : dict
            isochrone input
        model : str
            stellar evolution model grid (default=mist)
        maxAV : float
            maximum extinction [mag]
        max_distance : float
            maximum distance [pc]
        binary_star : bool
            use binary star model if True else False (default=False)
        Returns
        -------
        isochrones_model
        """
        try:
            from isochrones import (
                get_ichrone,
                SingleStarModel,
                BinaryStarModel,
            )
        except Exception:
            cmd = "pip install isochrones\n"
            cmd = "You may want to also install pymultinest for Nested Sampling.\n"
            cmd = "See https://github.com/JohannesBuchner/PyMultiNest"
            raise ModuleNotFoundError(cmd)

        mist = get_ichrone(model, bands=bands)
        self.mist = mist
        iso_params = (
            self.get_iso_params() if iso_params is None else iso_params
        )
        if self.verbose:
            print(iso_params)
        if binary_star:
            model = BinaryStarModel
        else:
            model = SingleStarModel
        self.isochrones_model = model(
            self.mist,
            maxAV=maxAV,
            max_distance=max_distance,
            ra=self.target_coord.ra.deg,
            dec=self.target_coord.dec.deg,
            name=self.target_name,
            **iso_params,
        )
        # set mass upper limit up to 10 Msol
        self.isochrones_model.set_bounds(mass=(0.1, 10))
        # set eep upper limit up to asymptotic giant branch
        self.isochrones_model.set_bounds(eep=(0, 808))
        return self.isochrones_model
    
    def run_isochrones(
        self, iso_params=None, binary_star=False, overwrite=False, **kwargs
    ):
        """
        Parameters
        ----------
        iso_params : dict
            isochrone input
        binary_star : bool
            use binary star model if True else False (default=False)
        overwrite : bool
            re-run isochrones from scratch
        Returns
        -------
        isochrones_model
        Note:
        * Use `init_isochones` for detailed isochrones model initialization.
        https://isochrones.readthedocs.io/en/latest/quickstart.html#Fit-physical-parameters-of-a-star-to-observed-data

        * See mod._priors for priors; for multi-star systems, see
        https://isochrones.readthedocs.io/en/latest/multiple.html
        FIXME: nsteps param in mod.fit() cannot be changed
        """
        # Create a dictionary of observables
        if self.mist is None:
            iso_params = (
                self.get_iso_params() if iso_params is None else iso_params
            )
            self.init_isochrones(iso_params=iso_params)
        else:
            print("Using previously initialized model.")

        model = self.isochrones_model
        if model._samples is not None:
            if not overwrite:
                if self.verbose:
                    print(
                        "Loading previous samples. Otherwise, try overwrite=True."
                    )
            else:
                if self.verbose:
                    print("Overwriting previous run.")
        if model.use_emcee:
            print("Method: Affine-invariant MCMC")
            # kwargs = {"niter": int(nsteps)}
        else:
            print("Method: Nested Sampling")
            # kwargs = {"n_live_points": int(nsteps)}
        # fit
        try:
            logprior0 = model.lnprior(self.iso_params0)
            loglike0 = model.lnlike(self.iso_params0)
            logpost0 = model.lnpost(self.iso_params0)
            msg = "Initial values:\n"
            msg += "logpost=loglike+logprior = "
            msg += f"{loglike0:.2f} + {logprior0:.2f} = {logpost0:.2f}"
            if self.verbose:
                print(msg)
        except Exception as e:
            errmsg = f"Error: {e}\n"
            errmsg += "Error in calculating logprior. Check `iso_params` input values."
            raise ValueError(errmsg)

        # nsteps = nsteps if nsteps is not None else self.mcmc_steps
        model.fit(overwrite=overwrite, **kwargs)

        # Note: median!=MAP
        # iso_params0_ = model.samples.median().values
        iso_params0_ = model.map_pars
        logprior = model.lnprior(iso_params0_)
        loglike = model.lnlike(iso_params0_)
        logpost = model.lnpost(iso_params0_)
        msg = "Final values:\n"
        msg += "logpost=loglike+logprior = "
        msg += f"{loglike:.2f} + {logprior:.2f} = {logpost:.2f}"
        if self.verbose:
            print(msg)
        if not model.use_emcee:
            print(f"Model evidence: {model.evidence}")
        return model

    def get_isochrones_prior_samples(self, nsamples=int(1e4)):
        """sample default priors

        Returns dataframe
        """
        model = self.isochrones_model
        errmsg = "self.run_isochrones"
        assert model is not None, errmsg
        samples = {}
        for param in model._priors:
            if param == "eep":
                age, feh = self.iso_params0[1], self.iso_params0[2]
                samples[param] = model._priors[param].sample(
                    nsamples, age=age, feh=feh
                )
            else:
                samples[param] = model._priors[param].sample(nsamples)
        return pd.DataFrame(samples)

    def plot_isochrones_priors(self, kind="kde"):
        """plot default priors

        TODO: add units and prior name
        ChabrierPrior: LogNormalPrior+PowerLawPrior
        FehPrior: feh PDF based on local SDSS distribution
        AgePrior: FlatLogPrior, log10(age)
        DistancePrior: PowerLawPrior
        AVPrior: FlatPrior
        EEP_prior: BoundedPrior (See self.mist_eep_table)
        """

        fig, axs = pl.subplots(2, 3, figsize=(8, 8), constrained_layout=True)
        ax = axs.flatten()

        df = self.get_isochrones_prior_samples()
        for i, col in enumerate(df.columns):
            _ = df[col].plot(kind=kind, ax=ax[i])
            ax[i].set_title(col)
            xlims = self.isochrones_model._priors[col].bounds
            if i not in [0, 3]:
                ax[i].set_ylabel("")
            if np.isfinite(xlims).any():
                ax[i].set_xlim(xlims)
        fig.suptitle("Priors")
        # fig.subplots_adjust(wspace=0.1)
        return fig

    def plot_posterior_eep(self):
        """
        """
        errmsg = "try self.run_isochrones()"
        assert self.isochrones_model._samples is not None, errmsg
        emin = self.isochrones_model.derived_samples.eep.min() - 100
        emax = self.isochrones_model.derived_samples.eep.max() + 100

        idx = self.mist_eep_table["EEP Number"].between(emin, emax)
        tab = self.mist_eep_table.loc[idx, ["EEP Number", "Phase"]]

        # plot kde
        ax = self.isochrones_model.derived_samples.eep.plot(kind="kde")
        n = 1
        for _, row in tab.iterrows():
            ax.axvline(
                row["EEP Number"], 0, 1, label=row["Phase"], ls="--", c=f"C{n}"
            )
            n += 1
        ax.set_xlabel("Equal Evolutionary Point")
        ax.set_title(self.target_name)
        ax.legend()
        return ax

    # @classmethod
    def get_isochrones_results(self):
        if self.isochrones_model is not None:
            return self.isochrones_model.derived_samples
        else:
            raise ValueError("Try self.run_isochrones()")

    # @classmethod
    def get_isochrones_results_summary(self):
        if self.isochrones_model is not None:
            return self.isochrones_model.derived_samples.describe()
        else:
            raise ValueError("Try self.run_isochrones()")
            
    def plot_isochrones_priors(self, kind="kde"):
        """plot default priors

        TODO: add units and prior name
        ChabrierPrior: LogNormalPrior+PowerLawPrior
        FehPrior: feh PDF based on local SDSS distribution
        AgePrior: FlatLogPrior, log10(age)
        DistancePrior: PowerLawPrior
        AVPrior: FlatPrior
        EEP_prior: BoundedPrior (See self.mist_eep_table)
        """

        fig, axs = pl.subplots(2, 3, figsize=(8, 8), constrained_layout=True)
        ax = axs.flatten()

        df = self.get_isochrones_prior_samples()
        for i, col in enumerate(df.columns):
            _ = df[col].plot(kind=kind, ax=ax[i])
            ax[i].set_title(col)
            xlims = self.isochrones_model._priors[col].bounds
            if i not in [0, 3]:
                ax[i].set_ylabel("")
            if np.isfinite(xlims).any():
                ax[i].set_xlim(xlims)
        fig.suptitle("Priors")
        # fig.subplots_adjust(wspace=0.1)
        return fig
    
    def get_isochrones_prior_samples(self, nsamples=int(1e4)):
        """sample default priors

        Returns dataframe
        """
        model = self.isochrones_model
        errmsg = "self.run_isochrones"
        assert model is not None, errmsg
        samples = {}
        for param in model._priors:
            if param == "eep":
                age, feh = self.iso_params0[1], self.iso_params0[2]
                samples[param] = model._priors[param].sample(
                    nsamples, age=age, feh=feh
                )
            else:
                samples[param] = model._priors[param].sample(nsamples)
        return pd.DataFrame(samples)
    
    def plot_flatchain(self, burnin=None):
        """
        useful to estimate burn-in
        """
        if self.stardate is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.stardate

        chain = star.sampler.chain
        nwalkers, nsteps, ndim = chain.shape
        fig, axs = pl.subplots(ndim, 1, figsize=(15, ndim), sharex=True)
        [
            axs.flat[i].plot(
                c, drawstyle="steps", color="k", alpha=4.0 / nwalkers
            )
            for i, c in enumerate(chain.T)
        ]
        [axs.flat[i].set_ylabel(l) for i, l in enumerate(self.iso_param_names)]
        return fig

    def plot_corner(
        self,
        posterior="physical",
        use_isochrones=True,
        columns=None,
        burnin=None,
        thin=None,
    ):
        """
        use_isochrones : bool
            use isochrones or stardate results
        posterior : str
            'observed', 'physical', 'derived'
        columns : list
            columns to plot if use_isochrones=True and posterior='derived'
        See https://isochrones.readthedocs.io/en/latest/starmodel.html
        """
        try:
            from corner import corner
        except Exception:
            raise ValueError("pip install corner")

        errmsg = "Try run_isochrones() or run_stardate()"
        assert (self.isochrones_model is not None) | (
            self.stardate is not None
        ), errmsg

        burnin = burnin if burnin is not None else self.burnin
        thin = thin if thin is not None else self.thin

        if use_isochrones:
            if self.isochrones_model is None:
                raise ValueError("Try self.run_isochrones()")
            else:
                star = self.isochrones_model

            if posterior == "observed":
                fig = star.corner_observed()
                # columns = star.observed_quantities
                # data = star._samples
                # fig = corner(data, labels=columns,
                #        quantiles=[0.16, 0.5, 0.84],
                #        truth_color='C1',
                #        show_titles=True, title_kwargs={"fontsize": 12})
            elif posterior == "physical":
                # fig = star.corner_physical()
                columns = star.physical_quantities
                data = star._derived_samples[columns]
                fig = corner(
                    data,
                    labels=columns,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_kwargs={"fontsize": 12},
                )
            elif posterior == "derived":
                if columns is None:
                    print("Supply any columns:")
                    print(star.derived_samples.columns)
                    return None
                else:
                    # fig = star.corner_derived(columns)
                    data = star._derived_samples[columns]
                    fig = corner(
                        data,
                        labels=columns,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12},
                    )
            else:
                raise ValueError("Use posterior=(observed,physical,derived)")

        else:
            if self.stardate is None:
                raise ValueError("Try self.run_stardate()")
            else:
                star = self.stardate

            chain = star.sampler.chain
            nwalkers, nsteps, ndim = chain.shape
            samples = chain[:, burnin::thin, :].reshape((-1, ndim))

            from isochrones.mist import MIST_Isochrone

            mist = MIST_Isochrone()
            # samples needed to interpolate parameters in mist isochrones
            eep_samples = samples[:, 0]
            log_age_samples = samples[:, 1]
            feh_samples = samples[:, 2]

            if posterior == "observed":
                # fig = corner(samples, labels=self.iso_param_names)
                columns = self.iso_param_names
                fig = corner(
                    samples,
                    labels=columns,
                    quantiles=[0.16, 0.5, 0.84],
                    truth_color="C1",
                    show_titles=True,
                    title_kwargs={"fontsize": 12},
                )

            elif posterior == "physical":
                columns = [
                    "mass",
                    "radius",
                    "age",
                    "Teff",
                    "logg",
                    "feh",
                ]  # , 'distance', 'AV'
                derived_samples = mist.interp_value(
                    [eep_samples, log_age_samples, feh_samples], columns
                )
                fig = corner(
                    derived_samples,
                    labels=columns,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_kwargs={"fontsize": 12},
                )

            elif posterior == "derived":
                avail_columns = [
                    "eep",
                    "age",
                    "feh",
                    "mass",
                    "initial_mass",
                    "radius",
                    "density",
                    "logTeff",
                    "Teff",
                    "logg",
                    "logL",
                    "Mbol",
                    "dm_deep"
                    # 'delta_nu', 'nu_max', 'phase',
                    # 'J_mag', 'H_mag', 'K_mag', 'G_mag', 'BP_mag', 'RP_mag',
                    # 'W1_mag', 'W2_mag', 'W3_mag', 'TESS_mag', 'Kepler_mag',
                    # 'parallax', 'distance', 'AV'
                ]
                if columns is None:
                    print("Supply any columns:")
                    print(avail_columns)
                    return None
                else:
                    derived_samples = mist.interp_value(
                        [eep_samples, log_age_samples, feh_samples], columns
                    )
                    fig = corner(
                        derived_samples,
                        labels=columns,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12},
                    )
        return fig
