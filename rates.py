"""
Diferencial de tasas Chile − EE.UU. histórico, para modelar el SWAP/rollover.

Mantener una posición de noche cuesta/paga el diferencial de tasas:
  - Largo USD/CLP  (corto peso): pagas la tasa chilena, recibes la de EE.UU. → normalmente PAGAS.
  - Corto USD/CLP  (largo peso): recibes la chilena, pagas la de EE.UU. → normalmente COBRAS.

Datos: TPM de Chile (mindicador.cl, por año) y tasa corta de EE.UU. (^IRX, Yahoo).
"""
import datetime as dt
import numpy as np
import requests

UA = {"User-Agent": "Mozilla/5.0"}


def _tpm_chile(años):
    d = {}
    for y in años:
        try:
            r = requests.get(f"https://mindicador.cl/api/tpm/{y}", headers=UA, timeout=20)
            for x in r.json()["serie"]:
                d[x["fecha"][:10]] = float(x["valor"])
        except Exception:
            pass
    return d


def _us_short():
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/^IRX",
                             params={"interval": "1d", "range": "5y"}, headers=UA, timeout=25)
            res = r.json()["chart"]["result"][0]
            ts, c = res["timestamp"], res["indicators"]["quote"][0]["close"]
            return {dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d"): c[i]
                    for i in range(len(ts)) if c[i] is not None}
        except Exception:
            continue
    return {}


def _ffill(d, fechas):
    """Rellena hacia adelante una serie {fecha:valor} sobre la lista `fechas`."""
    out = np.zeros(len(fechas)); ult = None
    for i, f in enumerate(fechas):
        if f in d:
            ult = d[f]
        out[i] = ult if ult is not None else (next(iter(d.values())) if d else 0.0)
    return out


def diferencial(com):
    """Devuelve array (cl_rate − us_rate) en % anual, alineado a las fechas `com`."""
    años = sorted(set(int(f[:4]) for f in com))
    tpm = _tpm_chile(años)
    us = _us_short()
    cl_arr = _ffill(tpm, com)
    us_arr = _ffill(us, com)
    return cl_arr - us_arr, cl_arr, us_arr
