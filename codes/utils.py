# -*- coding: utf-8 -*-

import itertools
from os import makedirs
from os.path import join, exists
from pathlib import Path

import numpy as np
import pandas as pd
import lightkurve as lk
from astropy import units as u
from astropy.coordinates import SkyCoord, Distance
from astroquery.mast import Observations, Catalogs
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

DATA_PATH = '../data'

def flatten_list(lol):
    """flatten list of list (lol)"""
    return list(itertools.chain.from_iterable(lol))

def get_coord_from_epicid(epicid):
    """Scrape exofop website for relevant info"""
    r = Simbad.query_object(f"EPIC {epicid}")
    if r is not None:
        r = r.to_pandas()
        ra, dec = r["RA"].values[0], r["DEC"].values[0]
    else:
        errmsg = f"Simbad query result is empty for EPIC {epicid}.\n"
        url = f"https://exofop.ipac.caltech.edu/k2/edit_target.php?id={epicid}"
        errmsg += f"Using target coordinates from:\n{url}"
        print(errmsg)
        # raise ValueError(errmsg)
        res = pd.read_html(url)
        if len(res) > 1:
            r = res[4].loc[1]
            ra, dec = r[1], r[2]
        else:
            raise ValueError(f"EPIC {epicid} does not exist.")
    coord = SkyCoord(ra=ra, dec=dec, unit=("hourangle", "deg"))
    return coord

def get_epicid_from_k2name(k2name):
    res = lk.search_targetpixelfile(k2name, mission="K2")
    target_name = res.table.to_pandas().target_name[0]
    epicid = int(target_name[4:])  # skip ktwo
    return epicid

def get_mag_err_from_flux(flux, flux_err):
    """
    equal to 1.086/(S/N)
    """
    return 2.5 * np.log10(1 + flux_err / flux)

def get_target_coord(
    ra=None,
    dec=None,
    toi=None,
    ctoi=None,
    tic=None,
    epic=None,
    gaiaid=None,
    name=None,
):
    """get target coordinate
    """
    if np.all([ra, dec]):
        target_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    elif toi:
        toi_params = get_toi(toi=toi, clobber=False, verbose=False)
        target_coord = SkyCoord(
            ra=toi_params["RA"].values[0],
            dec=toi_params["Dec"].values[0],
            distance=toi_params["Stellar Distance (pc)"].values[0],
            unit=(u.hourangle, u.degree, u.pc),
        )
    elif ctoi:
        ctoi_params = get_ctoi(ctoi=ctoi, clobber=False, verbose=False)
        target_coord = SkyCoord(
            ra=ctoi_params["RA"].values[0],
            dec=ctoi_params["Dec"].values[0],
            distance=ctoi_params["Stellar Distance (pc)"].values[0],
            unit=(u.degree, u.degree, u.pc),
        )
    elif tic:
        df = Catalogs.query_criteria(catalog="Tic", ID=tic).to_pandas()
        target_coord = SkyCoord(
            ra=df.iloc[0]["ra"],
            dec=df.iloc[0]["dec"],
            distance=Distance(parallax=df.iloc[0]["plx"] * u.mas).pc,
            unit=(u.degree, u.degree, u.pc),
        )
    # name resolver
    elif epic is not None:
        target_coord = get_coord_from_epicid(epic)
    elif gaiaid is not None:
        target_coord = SkyCoord.from_name(f"Gaia DR2 {gaiaid}")
    elif name is not None:
        target_coord = SkyCoord.from_name(name)
    else:
        raise ValueError("Supply RA & Dec, TOI, TIC, or Name")
    return target_coord

def get_mist_eep_table():
    """
    For eep phases, see
    http://waps.cfa.harvard.edu/MIST/README_tables.pdf
    """
    fp = Path('../data', "mist_eep_table.csv")
    return pd.read_csv(fp, comment="#")

def map_float(x):
    return list(map(float, x))

def get_nexsci_archive(table="all"):
    """
    direct download from NExSci archive
    """
    base_url = "https://exoplanetarchive.ipac.caltech.edu/"
    settings = "cgi-bin/nstedAPI/nph-nstedAPI?table="
    if table == "all":
        url = base_url + settings + "exomultpars"
    elif table == "confirmed":
        url = base_url + settings + "exoplanets"
    elif table == "composite":
        url = base_url + settings + "compositepars"
    else:
        raise ValueError("table=[all, confirmed, composite]")
    df = pd.read_csv(url)
    return df


def get_nexsci_candidates(cache=False):
    """
    """
    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
    except Exception:
        raise ModuleNotFoundError("pip install astroquery --update")
    candidates = NasaExoplanetArchive.query_criteria(
        table="k2candidates", cache=cache
    )
    nexsci_pc = candidates.to_pandas()
    # nexsci_pc = nexsci_pc.query("k2c_disp=='CONFIRMED'")
    return nexsci_pc.query("k2c_disp=='CANDIDATE'")


def get_vizier_tables(key, tab_index=None, row_limit=50, verbose=True):
    """
    Parameters
    ----------
    key : str
        vizier catalog key
    tab_index : int
        table index to download and parse
    Returns
    -------
    tables if tab_index is None else parsed df
    """
    if row_limit == -1:
        msg = "Downloading all tables in "
    else:
        msg = f"Downloading the first {row_limit} rows of each table in "
    msg += f"{key} from vizier."
    if verbose:
        print(msg)
    # set row limit
    Vizier.ROW_LIMIT = row_limit

    tables = Vizier.get_catalogs(key)
    errmsg = "No data returned from Vizier."
    assert tables is not None, errmsg

    if tab_index is None:
        if verbose:
            print({k: tables[k]._meta["description"] for k in tables.keys()})
        return tables
    else:
        df = tables[tab_index].to_pandas()
        df = df.applymap(
            lambda x: x.decode("ascii") if isinstance(x, bytes) else x
        )
        return df


def get_tois_mass_RV_K(clobber=False):
    fp = Path(DATA_PATH, "TOIs2.csv")
    if clobber:
        try:
            from mrexo import predict_from_measurement, generate_lookup_table
        except Exception:
            raise ModuleNotFoundError("pip install mrexo")

        tois = get_tois()

        masses = {}
        for key, row in tqdm(tois.iterrows()):
            toi = row["TOI"]
            Rp = row["Planet Radius (R_Earth)"]
            Rp_err = row["Planet Radius (R_Earth) err"]
            Mp, (Mp_lo, Mp_hi), iron_planet = predict_from_measurement(
                measurement=Rp,
                measurement_sigma=Rp_err,
                qtl=[0.16, 0.84],
                dataset="kepler",
            )
            masses[toi] = (Mp, Mp_lo, Mp_hi)

        df = pd.DataFrame(masses).T
        df.columns = [
            "Planet mass (Mp_Earth)",
            "Planet mass (Mp_Earth) lo",
            "Planet mass (Mp_Earth) hi",
        ]
        df.index.name = "TOI"
        df = df.reset_index()

        # df["RV_K_lo"] = get_RV_K(
        #     tois["Period (days)"],
        #     tois["Stellar Radius (R_Sun)"],  # should be Mstar
        #     df["Planet mass (Mp_Earth) lo"],
        #     with_unit=True,
        # )

        # df["RV_K_hi"] = get_RV_K(
        #     tois["Period (days)"],
        #     tois["Stellar Radius (R_Sun)"],  # should be Mstar
        #     df["Planet mass (Mp_Earth) hi"],
        #     with_unit=True,
        # )

        joint = pd.merge(tois, df, on="TOI")
        joint.to_csv(fp, index=False)
        print(f"Saved: {fp}")
    else:
        joint = pd.read_csv(fp)
        print(f"Loaded: {fp}")
    return joint


def get_tois(
    clobber=True,
    outdir=DATA_PATH,
    verbose=False,
    remove_FP=True,
    remove_known_planets=False,
    add_FPP=False,
):
    """Download TOI list from TESS Alert/TOI Release.

    Parameters
    ----------
    clobber : bool
        re-download table and save as csv file
    outdir : str
        download directory location
    verbose : bool
        print texts

    Returns
    -------
    d : pandas.DataFrame
        TOI table as dataframe
    """
    dl_link = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    fp = join(outdir, "TOIs.csv")
    if not exists(outdir):
        makedirs(outdir)

    if not exists(fp) or clobber:
        d = pd.read_csv(dl_link)  # , dtype={'RA': float, 'Dec': float})
        msg = f"Downloading {dl_link}\n"
        if add_FPP:
            fp2 = join(outdir, "Giacalone2020/tab4.txt")
            classified = ascii.read(fp2).to_pandas()
            fp3 = join(outdir, "Giacalone2020/tab5.txt")
            unclassified = ascii.read(fp3).to_pandas()
            fpp = pd.concat(
                [
                    classified[["TOI", "FPP-2m", "FPP-30m"]],
                    unclassified[["TOI", "FPP"]],
                ],
                sort=True,
            )
            d = pd.merge(d, fpp, how="outer").drop_duplicates()
        d.to_csv(fp, index=False)
    else:
        d = pd.read_csv(fp).drop_duplicates()
        msg = f"Loaded: {fp}\n"
    assert len(d) > 1000, f"{fp} likely has been overwritten!"

    # remove False Positives
    if remove_FP:
        d = d[d["TFOPWG Disposition"] != "FP"]
        msg += "TOIs with TFPWG disposition==FP are removed.\n"
    if remove_known_planets:
        planet_keys = [
            "HD",
            "GJ",
            "LHS",
            "XO",
            "Pi Men" "WASP",
            "SWASP",
            "HAT",
            "HATS",
            "KELT",
            "TrES",
            "QATAR",
            "CoRoT",
            "K2",  # , "EPIC"
            "Kepler",  # "KOI"
        ]
        keys = []
        for key in planet_keys:
            idx = ~np.array(
                d["Comments"].str.contains(key).tolist(), dtype=bool
            )
            d = d[idx]
            if idx.sum() > 0:
                keys.append(key)
        msg += f"{keys} planets are removed.\n"
    msg += f"Saved: {fp}\n"
    if verbose:
        print(msg)
    return d.sort_values("TOI")


def get_toi(toi, verbose=False, remove_FP=True, clobber=False):
    """Query TOI from TOI list

    Parameters
    ----------
    toi : float
        TOI id
    clobber : bool
        re-download csv file
    outdir : str
        csv path
    verbose : bool
        print texts

    Returns
    -------
    q : pandas.DataFrame
        TOI match else None
    """
    df = get_tois(verbose=False, remove_FP=remove_FP, clobber=clobber)

    if isinstance(toi, int):
        toi = float(str(toi) + ".01")
    else:
        planet = str(toi).split(".")[1]
        assert len(planet) == 2, "use pattern: TOI.01"
    idx = df["TOI"].isin([toi])
    q = df.loc[idx]
    assert len(q) > 0, "TOI not found!"

    q.index = q["TOI"].values
    if verbose:
        print("Data from TOI Release:\n")
        columns = [
            "Period (days)",
            "Epoch (BJD)",
            "Duration (hours)",
            "Depth (ppm)",
            "Comments",
        ]
        print(f"{q[columns].T}\n")

    if q["TFOPWG Disposition"].isin(["FP"]).any():
        print("\nTFOPWG disposition is a False Positive!\n")

    return q.sort_values(by="TOI", ascending=True)


def get_ctois(clobber=True, outdir=DATA_PATH, verbose=False, remove_FP=True):
    """Download Community TOI list from exofop/TESS.

    Parameters
    ----------
    clobber : bool
        re-download table and save as csv file
    outdir : str
        download directory location
    verbose : bool
        print texts

    Returns
    -------
    d : pandas.DataFrame
        CTOI table as dataframe

    See interface: https://exofop.ipac.caltech.edu/tess/view_ctoi.php
    See also: https://exofop.ipac.caltech.edu/tess/ctoi_help.php
    """
    dl_link = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
    fp = join(outdir, "CTOIs.csv")
    if not exists(outdir):
        makedirs(outdir)

    if not exists(fp) or clobber:
        d = pd.read_csv(dl_link)  # , dtype={'RA': float, 'Dec': float})
        msg = "Downloading {}\n".format(dl_link)
    else:
        d = pd.read_csv(fp).drop_duplicates()
        msg = "Loaded: {}\n".format(fp)
    d.to_csv(fp, index=False)

    # remove False Positives
    if remove_FP:
        d = d[d["User Disposition"] != "FP"]
        msg += "CTOIs with user disposition==FP are removed.\n"
    msg += "Saved: {}\n".format(fp)
    if verbose:
        print(msg)
    return d.sort_values("CTOI")


def get_ctoi(ctoi, verbose=False, remove_FP=False, clobber=False):
    """Query CTOI from CTOI list

    Parameters
    ----------
    ctoi : float
        CTOI id

    Returns
    -------
    q : pandas.DataFrame
        CTOI match else None
    """
    ctoi = float(ctoi)
    df = get_ctois(verbose=False, remove_FP=remove_FP, clobber=clobber)

    if isinstance(ctoi, int):
        ctoi = float(str(ctoi) + ".01")
    else:
        planet = str(ctoi).split(".")[1]
        assert len(planet) == 2, "use pattern: CTOI.01"
    idx = df["CTOI"].isin([ctoi])

    q = df.loc[idx]
    assert len(q) > 0, "CTOI not found!"

    q.index = q["CTOI"].values
    if verbose:
        print("Data from CTOI Release:\n")
        columns = [
            "Period (days)",
            "Midpoint (BJD)",
            "Duration (hours)",
            "Depth ppm",
            "Notes",
        ]
        print(f"{q[columns].T}\n")
    if (q["TFOPWG Disposition"].isin(["FP"]).any()) | (
        q["User Disposition"].isin(["FP"]).any()
    ):
        print("\nTFOPWG/User disposition is a False Positive!\n")

    return q.sort_values(by="CTOI", ascending=True)


def get_specs_table_from_tfop(clobber=True, outdir=DATA_PATH, verbose=True):
    """
    html:
    https://exofop.ipac.caltech.edu/tess/view_spect.php?sort=id&ipp1=1000

    plot notes:
    https://exofop.ipac.caltech.edu/tess/classification_plots.php
    """
    base = "https://exofop.ipac.caltech.edu/tess/"
    fp = join(outdir, "tfop_sg2_spec_table.csv")
    if not exists(fp) or clobber:
        url = base + "download_spect.php?sort=id&output=csv"
        df = pd.read_csv(url)
        df.to_csv(fp, index=False)
        if verbose:
            print(f"Saved: {fp}")
    else:
        df = pd.read_csv(fp)
        if verbose:
            print(f"Loaded: {fp}")
    return df

