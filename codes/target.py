# -*- coding: utf-8 -*-

r"""
Module for star bookkeeping, e.g. position, catalog cross-matching, archival data look-up.
"""

# Import standard library
# from inspect import signature
from pathlib import Path
import warnings
from pprint import pprint
import logging

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.mast import Observations, Catalogs
from astropy.coordinates import SkyCoord, Distance
import astropy.units as u
from tqdm import tqdm

from utils import flatten_list, get_epicid_from_k2name, get_toi, get_tois, get_target_coord, get_k2_data_from_exofop

class Target(object):
    """
    Performs target resolution basic catalog cross-matching and archival data look-up
    """

    def __init__(
        self,
        name=None,
        toiid=None,
        ctoiid=None,
        ticid=None,
        epicid=None,
        gaiaDR2id=None,
        ra_deg=None,
        dec_deg=None,
        mission="tess",
        search_radius=30,
        verbose=True,
        clobber=False,
        check_if_variable=False,
    ):
        """
        Attributes
        ----------
        search_radius : float
            search radius for matching [arcsec]
        """
        self.clobber = clobber
        self.verbose = verbose
        self.mission = mission.lower()
        self.toi_params = None
        self.nea_params = None
        self.toiid = toiid if toiid is None else int(toiid)  # e.g. 837
        self.ctoiid = ctoiid  # e.g. 364107753.01
        self.ticid = ticid if ticid is None else int(ticid)  # e.g. 364107753
        self.epicid = epicid if epicid is None else int(epicid)  # 201270176
        self.gaiaid = gaiaDR2id  # e.g. Gaia DR2 5251470948229949568
        self.ra = ra_deg
        self.dec = dec_deg
        self.target_name = name  # e.g. Pi Mensae
        # determine target name
        if self.toiid is not None:
            name = f"TOI {self.toiid}"
        elif self.ticid is not None:
            name = f"TIC {self.ticid}"
        elif self.epicid is not None:
            name = f"EPIC {self.epicid}"
            self.mission = "k2"
        elif self.gaiaid is not None:
            name = f"Gaia DR2 {self.gaiaid}"
        elif self.target_name is not None:
            if self.target_name[:2].lower() == "k2":
                name = self.target_name.upper()
                self.mission = "k2"
                self.epicid = get_epicid_from_k2name(name)
            elif self.target_name[:6].lower() == "kepler":
                name = self.target_name.title()
                self.mission = "kepler"
            elif self.target_name[:4].lower() == "gaia":
                name = self.target_name.upper()
                if gaiaDR2id is None:
                    self.gaiaid = int(name.strip()[4:])
        elif (self.ra is not None) & (self.dec is not None):
            name = f"({self.ra:.3f}, {self.dec:.3f})"
        # specify name
        if self.target_name is None:
            self.target_name = name
        # check if TIC is a TOI
        if (self.ticid is not None) and (self.toiid is None):
            tois = get_tois(clobber=True, verbose=False)
            idx = tois["TIC ID"].isin([self.ticid])
            if sum(idx) > 0:
                self.toiid = tois.loc[idx, "TOI"].values[0]
                if self.verbose:
                    print(f"TIC {self.ticid} is TOI {int(self.toiid)}!")
        # query TOI params
        if self.toiid is not None:
            self.toi_params = get_toi(
                toi=self.toiid, clobber=self.clobber, verbose=False
            ).iloc[0]
            # nplanets = int(self.toi_params["Planet Num"])
            # if nplanets > 1:
            #     print(f"Target has {nplanets} planets.")
        if (self.ticid is None) and (self.toiid is not None):
            self.ticid = int(self.toi_params["TIC ID"])
        # get coordinates
        self.target_coord = get_target_coord(
            ra=self.ra,
            dec=self.dec,
            toi=self.toiid,
            ctoi=self.ctoiid,
            tic=self.ticid,
            epic=self.epicid,
            gaiaid=self.gaiaid,
            name=self.target_name,
        )

        self.search_radius = search_radius * u.arcsec
        self.tic_params = None
        self.gaia_params = None
        self.gaia_sources = None
        self.gmag = None
        self.distance_to_nearest_cluster_member = None
        self.nearest_cluster_member = None
        self.nearest_cluster_members = None
        self.nearest_cluster_name = None
        self.vizier_tables = None
        self.cc = None
        # as opposed to self.cc.all_clusters, all_clusters has uncertainties
        # appended in get_cluster_membership
        self.all_clusters = None
        self.harps_bank_table = None
        self.harps_bank_rv = None
        self.harps_bank_target_name = None
        self.variable_star = False
        if self.verbose:
            print(f"Target: {name}")
        if check_if_variable:
            self.query_variable_star_catalogs()

    def __repr__(self):
        """Override to print a readable string representation of class
        """
        # params = signature(self.__init__).parameters
        # val = repr(getattr(self, key))

        included_args = [
            # ===target attributes===
            "name",
            "toiid",
            "ctoiid",
            "ticid",
            "epicid",
            "gaiaDR2id",
            "ra_deg",
            "dec_deg",
            "target_coord",
            "search_radius",
            "mission",
            "campaign",
            "all_sectors",
            "all_campaigns",
            # ===tpf attributes===
            "sap_mask",
            "quality_bitmask",
            "calc_fpp",
            # 'aper_radius', 'threshold_sigma', 'percentile' #if sap_mask!='pipeline'
            # cutout_size #for FFI
            # ===lightcurve===
            "lctype",
            "aper_idx",
        ]
        args = []
        for key in self.__dict__.keys():
            val = self.__dict__.get(key)
            if key in included_args:
                if key == "target_coord":
                    # format coord
                    coord = self.target_coord.to_string("decimal")
                    args.append(f"{key}=({coord.replace(' ',',')})")
                elif val is not None:
                    args.append(f"{key}={val}")
        args = ", ".join(args)
        return f"{type(self).__name__}({args})"

    def query_variable_star_catalogs(self):
        """
        Check for variable star flag in vizier and var in catalog title
        """
        # tabs = self.query_vizier_param('var')
        # if len(tabs)>1:
        #     print(tabs)
        #     print("***Target has a variable star flag!***")
        #     self.variable_star = True
        all_tabs = self.query_vizier(verbose=False)

        keys = [
            "V/150/variabls",
            "J/AcA/66/421/ecl",
            "B/gcvs/gcvs_cat",
            "B/vsx/vsx",
            "J/AJ/156/234/table4",
            "J/MNRAS/488/4905/table2",
            "J/AJ/155/39/Variables",
        ]
        for n, tab in enumerate(all_tabs.keys()):
            for key in keys:
                if tab in key:
                    d = all_tabs[n].to_pandas().squeeze()
                    print(f"{key}:\n{d}")
                    self.variable_star = True

        # check for `var` in catalog title
        idx = [
            n if "var" in t._meta["description"] else False
            for n, t in enumerate(all_tabs)
        ]
        for i in idx:
            if i:
                tab = all_tabs[i]
                s = tab.to_pandas().squeeze().str.decode("ascii")
                print(f"\nSee also: {tab._meta['name']}\n{s}")
                self.variable_star = True

    def query_gaia_dr2_catalog(
        self, radius=None, return_nearest_xmatch=False, verbose=None
    ):
        """
        cross-match to Gaia DR2 catalog by angular separation
        position (accounting for proper motion) and brightess
        (comparing Tmag to Gmag whenever possible)

        Take caution:
        * phot_proc_mode=0 (i.e. “Gold” sources, see Riello et al. 2018)
        * astrometric_excess_noise_sig < 5
        * astrometric_gof_al < 20
        * astrometric_chi2_al
        * astrometric_n_good_obs_al
        * astrometric_primary_flag
        * duplicated source=0
        * visibility_periods_used
        * phot_variable_flag
        * flame_flags
        * priam_flags
        * phot_(flux)_excess_factor
        (a measure of the inconsistency between GBP, G, and GRP bands
        typically arising from binarity, crowdening and incomplete background
        modelling).

        Parameter
        ---------
        radius : float
            query radius in arcsec
        return_nearest_xmatch : bool
            return nearest single star if True else possibly more matches

        Returns
        -------
        tab : pandas.DataFrame
            table of star match(es)

        Notes:
        1. See column meaning here: https://mast.stsci.edu/api/v0/_c_a_o_mfields.html

        2. Gaia DR2 parallax has -0.08 mas offset (Stassun & Toress 2018,
        https://arxiv.org/pdf/1805.03526.pdf)

        3. quadratically add 0.1 mas to the uncertainty to account for systematics
        in the Gaia DR2 data (Luri+2018)

        4. Gmag has an uncertainty of 0.01 mag (Casagrande & VandenBerg 2018)

        From Carillo+2019:
        The sample with the low parallax errors i.e. 0 < f < 0.1,
        has distances derived from simply inverting the parallax

        Whereas, the sample with higher parallax errors i.e. f > 0.1
        has distances derived from a Bayesian analysis following Bailer-Jones (2015),
        where they use a weak distance prior (i.e. exponentially decreasing space
        density prior) that changes with Galactic latitude and longitude

        5. See also Gaia DR2 Cross-match for the celestial reference system (ICRS)
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_proc/ssec_cu3ast_proc_xmatch.html
        and
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_cali/ssec_cu3ast_cali_frame.html

        6. See https://github.com/tzdwi/TESS-Gaia and https://github.com/JohannesBuchner/nway
        and Salvato+2018 Appendix A for catalog matching problem: https://arxiv.org/pdf/1705.10711.pdf

        See also CDIPS gaia query:
        https://github.com/lgbouma/cdips/blob/master/cdips/utils/gaiaqueries.py

        See also bulk query:
        https://gea.esac.esa.int/archive-help/tutorials/python_cluster/index.html
        """
        radius = self.search_radius if radius is None else radius * u.arcsec
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            # silenced when verbose=False instead of None
            print(
                f"""Querying Gaia DR2 catalog for ra,dec=({self.target_coord.to_string()}) within {radius:.2f}."""
            )
        # load gaia params for all TOIs
        tab = Catalogs.query_region(
            self.target_coord, radius=radius, catalog="Gaia", version=2
        ).to_pandas()
        # rename distance to separation because it is confusing
        tab = tab.rename(columns={"distance": "separation"})
        # convert from arcmin to arcsec
        tab["separation"] = tab["separation"].apply(
            lambda x: x * u.arcmin.to(u.arcsec)
        )
        errmsg = f"No gaia star within {radius}. Use radius>{radius}"
        assert len(tab) > 0, errmsg
        tab["source_id"] = tab.source_id.astype(int)
        # check if results from DR2 (epoch 2015.5)
        assert np.all(
            tab["ref_epoch"].isin([2015.5])
        ), "Epoch not 2015 (version<2?)"

        if return_nearest_xmatch:
            nearest_match = tab.iloc[0]
            tplx = float(nearest_match["parallax"])
            if np.isnan(tplx) | (tplx < 0):
                print(f"Target parallax ({tplx} mas) is omitted!")
                tab["parallax"] = np.nan
        else:
            nstars = len(tab)
            idx1 = tab["parallax"] < 0
            tab.loc[idx1, "parallax"] = np.nan  # replace negative with nan
            idx2 = tab["parallax"].isnull()
            errmsg = f"No stars within radius={radius} have positive Gaia parallax!\n"
            if idx1.sum() > 0:
                errmsg += (
                    f"{idx1.sum()}/{nstars} stars have negative parallax!\n"
                )
            if idx2.sum() > 0:
                errmsg += f"{idx2.sum()}/{nstars} stars have no parallax!"
            assert len(tab) > 0, errmsg
        """
        FIXME: check parallax error here and apply corresponding distance calculation: see Note 1
        """
        if self.gaiaid is not None:
            errmsg = "Catalog does not contain target gaia id."
            assert np.any(tab["source_id"].isin([self.gaiaid])), errmsg

        # add gaia distance to target_coord
        # FIXME: https://docs.astropy.org/en/stable/coordinates/transforming.html
        gcoords = SkyCoord(
            ra=tab["ra"],
            dec=tab["dec"],
            unit="deg",
            frame="icrs",
            obstime="J2015.5",
        )
        # precess coordinate from Gaia DR2 epoch to J2000
        gcoords = gcoords.transform_to("icrs")
        if self.gaiaid is None:
            # find by nearest distance (for toiid or ticid input)
            idx = self.target_coord.separation(gcoords).argmin()
        else:
            # find by id match for gaiaDR2id input
            idx = tab.source_id.isin([self.gaiaid]).argmax()
        star = tab.loc[idx]
        # get distance from parallax
        if star["parallax"] > 0:
            target_dist = Distance(parallax=star["parallax"] * u.mas)
        else:
            target_dist = np.nan

        # redefine skycoord with coord and distance
        target_coord = SkyCoord(
            ra=self.target_coord.ra,
            dec=self.target_coord.dec,
            distance=target_dist,
        )
        self.target_coord = target_coord

        nsources = len(tab)
        if return_nearest_xmatch or (nsources == 1):
            if nsources > 1:
                print(f"There are {nsources} gaia sources within {radius}.")
            target = tab.iloc[0]
            if self.gaiaid is not None:
                id = int(target["source_id"])
                msg = f"Nearest match ({id}) != {self.gaiaid}"
                assert int(self.gaiaid) == id, msg
            else:
                self.gaiaid = int(target["source_id"])
            self.gaia_params = target
            self.gmag = target["phot_g_mean_mag"]
            ens = target["astrometric_excess_noise_sig"]
            if ens >= 5:
                msg = f"astrometric_excess_noise_sig={ens:.2f} (>5 hints binarity).\n"
                print(msg)
            gof = target["astrometric_gof_al"]
            if gof >= 20:
                msg = f"astrometric_gof_al={gof:.2f} (>20 hints binarity)."
                print(msg)
            if (ens >= 5) or (gof >= 20):
                print("See https://arxiv.org/pdf/1804.11082.pdf\n")
            delta = np.hypot(target["pmra"], target["pmdec"])
            if abs(delta) > 10:
                print("High proper-motion star:")
                print(
                    f"(pmra,pmdec)=({target['pmra']:.2f},{target['pmdec']:.2f}) mas/yr"
                )
            if target["visibility_periods_used"] < 6:
                msg = "visibility_periods_used<6 so no astrometric solution\n"
                msg += "See https://arxiv.org/pdf/1804.09378.pdf\n"
                print(msg)
            ruwe = list(self.query_vizier_param("ruwe").values())
            if len(ruwe) > 0 and ruwe[0] > 1.4:
                msg = f"RUWE={ruwe[0]:.1f}>1.4 means target is non-single or otherwise problematic for the astrometric solution."
                print(msg)
            return target  # return series of len 1
        else:
            # if self.verbose:
            #     d = self.get_nearby_gaia_sources()
            #     print(d)
            self.gaia_sources = tab
            return tab  # return dataframe of len 2 or more

    def query_tic_catalog(self, radius=None, return_nearest_xmatch=False):
        """
        Query TIC v8 catalog from MAST: https://astroquery.readthedocs.io/en/latest/mast/mast.html
        See column meaning in https://mast.stsci.edu/api/v0/_t_i_cfields.html
        and Table B in Stassun+2019: https://arxiv.org/pdf/1905.10694.pdf

        Parameter
        ---------
        radius : float
            query radius in arcsec

        Returns
        -------
        tab : pandas.DataFrame
            table of star match(es)
        """
        radius = self.search_radius if radius is None else radius * u.arcsec
        if self.verbose:
            print(
                f"Querying TIC catalog for ra,dec=({self.target_coord.to_string()}) within {radius}."
            )
        # NOTE: check tic version
        tab = Catalogs.query_region(
            self.target_coord, radius=radius, catalog="TIC"
        ).to_pandas()
        errmsg = f"No TIC star within {self.search_radius}"
        nsources = len(tab)
        assert nsources > 0, errmsg
        if return_nearest_xmatch or (nsources == 1):
            if nsources > 1:
                print(f"There are {nsources} TIC stars within {radius}")
            # get nearest match
            tab = tab.iloc[0]
            if tab.wdflag == 1:
                print("white dwarf flag = True!")
            if self.ticid is not None:
                id = int(tab["ID"])
                msg = f"Nearest match ({id}) != {self.ticid}"
                assert int(self.ticid) == id, msg
            else:
                if self.ticid is None:
                    self.ticid = int(tab["ID"])
        self.tic_params = tab
        return tab

    def validate_gaia_tic_xmatch(self, Rtol=0.3, mtol=0.5):
        """
        check if Rstar and parallax from 2 catalogs match,
        raises error otherwise
        """
        if (self.gaia_params is None) or (
            isinstance(self.gaia_params, pd.DataFrame)
        ):
            msg = "run query_gaia_dr2_catalog(return_nearest_xmatch=True)"
            raise ValueError(msg)
        g = self.gaia_params
        if (self.tic_params is None) or (
            isinstance(self.tic_params, pd.DataFrame)
        ):
            msg = "run query_tic_catalog(return_nearest_xmatch=True)"
            raise ValueError(msg)
        t = self.tic_params

        assert t.gaiaqflag==1, "TIC qflag!=1"

        # check magnitude
        if np.any(np.isnan([g.phot_g_mean_mag, t.Tmag])):
            msg = f"Gmag={g.phot_g_mean_mag}; Tmag={t.Tmag}"
            warnings.warn(msg)
            print(msg)
        else:
            assert np.allclose(g.phot_g_mean_mag, t.Tmag, rtol=mtol)

        # check parallax

        if np.any(np.isnan([g.parallax, t.plx])):
            msg = f"Gaia parallax={g.parallax}; TIC parallax={t.plx}"
            warnings.warn(msg)
            print(msg)
        else:
            assert np.allclose(g.parallax, t.plx, rtol=1e-3)

        # check Rstar
        if np.any(np.isnan([g.radius_val, t.rad])):
            msg = f"Gaia radius={g.radius_val}; TIC radius={t.rad}"
            warnings.warn(msg)
            print(msg)
        else:
            dradius = g.radius_val - t.rad
            msg = f"Rgaia-Rtic={g.radius_val:.2f}-{t.rad:.2f}={dradius:.2f}"
            assert np.allclose(g.radius_val, t.rad, rtol=Rtol), msg

        # check gaia ID
        if (self.gaiaid is not None) and (t["GAIA"] is not np.nan):
            assert g.source_id == int(t["GAIA"]), "Different source IDs!"

        print("Gaia and TIC catalog cross-match succeeded.")
        return True

    def query_simbad(self, radius=3):
        """
        Useful to get literature values for spectral type, Vsini, etc.
        See:
        https://astroquery.readthedocs.io/en/latest/simbad/simbad.html
        See also meaning of object types (otype) here:
        http://simbad.u-strasbg.fr/simbad/sim-display?data=otypes
        """
        radius = radius * u.arcsec if radius is not None else 3 * u.arcsec
        if self.verbose:
            print(
                f"Searching MAST for ({self.target_coord}) with radius={radius}."
            )
        simbad = Simbad()
        simbad.add_votable_fields("typed_id", "otype", "sptype", "rot", "mk")
        table = simbad.query_region(self.target_coord, radius=radius)
        if table is None:
            print("No result from Simbad.")
        else:
            df = table.to_pandas()
            df = df.drop(
                [
                    "RA_PREC",
                    "DEC_PREC",
                    "COO_ERR_MAJA",
                    "COO_ERR_MINA",
                    "COO_ERR_ANGLE",
                    "COO_QUAL",
                    "COO_WAVELENGTH",
                ],
                axis=1,
            )
            return df

    def query_vizier(self, radius=3, verbose=None):
        """
        Useful to get relevant catalogs from literature
        See:
        https://astroquery.readthedocs.io/en/latest/vizier/vizier.html
        """
        verbose = self.verbose if verbose is None else verbose
        radius = self.search_radius if radius is None else radius * u.arcsec
        if verbose:
            print(
                f"Searching Vizier: ({self.target_coord.to_string()}) with radius={radius}."
            )
        # standard column sorted in increasing distance
        v = Vizier(
            columns=["*", "+_r"],
            # column_filters={"Vmag":">10"},
            # keywords=['stars:white_dwarf']
        )
        if self.vizier_tables is None:
            tables = v.query_region(self.target_coord, radius=radius)
            if tables is None:
                print("No result from Vizier.")
            else:
                if verbose:
                    print(f"{len(tables)} tables found.")
                    pprint(
                        {
                            k: tables[k]._meta["description"]
                            for k in tables.keys()
                        }
                    )
                self.vizier_tables = tables
        else:
            tables = self.vizier_tables
        return tables

    def query_vizier_param(self, param=None, radius=3):
        """looks for value of param in each vizier table
        """
        if self.vizier_tables is None:
            tabs = self.query_vizier(radius=radius, verbose=False)
        else:
            tabs = self.vizier_tables

        if param is not None:
            idx = [param in i.columns for i in tabs]
            vals = {
                tabs.keys()[int(i)]: tabs[int(i)][param][0]
                for i in np.argwhere(idx).flatten()
            }
            if self.verbose:
                print(f"Found {sum(idx)} references in Vizier with `{param}`.")
            return vals
        else:
            cols = [i.to_pandas().columns.tolist() for i in tabs]
            print(f"Choose parameter:\n{list(np.unique(flatten_list(cols)))}")

    def query_vizier_mags(
        self,
        catalogs=["apass9", "gaiadr2", "2mass", "wise", "epic"],
        add_err=True,
    ):
        """
        TODO: use sedfitter
        """
        if self.vizier_tables is None:
            tabs = self.query_vizier(verbose=False)
        else:
            tabs = self.vizier_tables
        refs = {
            # "tycho": {"tabid": "I/259/tyc2", "cols": ["BTmag", "VTmag"]},
            "apass9": {"tabid": "II/336/apass9", "cols": ["Bmag", "Vmag"]},
            "gaiadr2": {
                "tabid": "I/345/gaia2",
                "cols": ["Gmag", "BPmag", "RPmag"],
            },
            "2mass": {"tabid": "II/246/out", "cols": ["Jmag", "Hmag", "Kmag"]},
            "wise": {
                "tabid": "II/328/allwise",
                "cols": ["W1mag", "W2mag", "W3mag", "W4mag"],
                "epic": {"tabid": "IV/34/epic", "cols": "Kpmag"},
            },
        }

        phot = []
        for cat in catalogs:
            if cat in refs.keys():
                tabid = refs[cat]["tabid"]
                cols = refs[cat]["cols"]
                if tabid in tabs.keys():
                    d = tabs[tabid].to_pandas()[cols]
                    phot.append(d)
                    if add_err:
                        ecols = ["e_" + col for col in refs[cat]["cols"]]
                        if cat != "tycho":
                            e = tabs[tabid].to_pandas()[ecols]
                            phot.append(e)
                else:
                    print(f"No {cat} data in vizier.")
        d = pd.concat(phot, axis=1).squeeze()
        d.name = self.target_name
        return d

    def get_k2_data_from_exofop(self, table="star"):
        """
        """
        return get_k2_data_from_exofop(self.epicid, table=table)

    @property
    def toi_Tmag(self):
        return None if self.toi_params is None else self.toi_params["TESS Mag"]

    @property
    def toi_Tmag_err(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["TESS Mag err"]
        )

    @property
    def toi_period(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Period (days)"]
        )

    @property
    def toi_epoch(self):
        return (
            None if self.toi_params is None else self.toi_params["Epoch (BJD)"]
        )

    @property
    def toi_duration(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Duration (hours)"]
        )

    @property
    def toi_period_err(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Period (days) err"]
        )

    @property
    def toi_epoch_err(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Epoch (BJD) err"]
        )

    @property
    def toi_duration_err(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Duration (hours) err"]
        )

    @property
    def toi_depth(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Depth (ppm)"] * 1e-6
        )

    @property
    def toi_depth_err(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Depth (ppm) err"] * 1e-6
        )
