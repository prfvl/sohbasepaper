from pathlib import Path
import numpy as np, pandas as pd, pickle

RAW = Path("../data/raw/alt")
OUT = Path("../data/processed")
GROUPS = ["regular_alt_batteries", "recommissioned_batteries", "second_life_batteries"]
COLS = ["time", "mode", "voltage_load", "current_load", "temperature_battery", "mission_type"]

def segments(df, min_rows=500):
    ref = (df["mission_type"] == 0) & (df["mode"] == -1)
    seg_id = (ref != ref.shift()).cumsum()
    out = []
    for _, g in df[ref].groupby(seg_id[ref]):
        if len(g) < min_rows:
            continue
        t = g["time"].to_numpy()
        i = np.abs(g["current_load"].to_numpy())
        q = np.concatenate([[0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t))]) / 3600
        out.append(dict(t=t, v=g["voltage_load"].to_numpy(), i=i,
                        temp=g["temperature_battery"].to_numpy(), q=q, Q=q[-1]))
    return out

def build():
    (OUT / "segments").mkdir(parents=True, exist_ok=True)
    rows = []
    for grp in GROUPS:
        for f in sorted((RAW / grp).glob("battery*.csv")):
            df = pd.read_csv(f, usecols=COLS)
            bad = df.apply(lambda c: pd.to_numeric(c, errors="coerce").isna() & c.notna()).sum()
            if bad.sum():
                print("  non-numeric values in", f.stem, bad[bad > 0].to_dict())
            df = df.apply(pd.to_numeric, errors="coerce")
            segs = segments(df)
            pickle.dump(segs, open(OUT / "segments" / f"{f.stem}.pkl", "wb"))
            for k, s in enumerate(segs):
                rows.append(dict(group=grp, pack=f.stem, seg=k, Q_Ah=s["Q"],
                                 t_start=s["t"][0], n_rows=len(s["t"]),
                                 v_min=np.nanmin(s["v"]), v_max=np.nanmax(s["v"]),
                                 temp_mean=np.nanmean(s["temp"])))
            print(grp, f.stem, len(segs), "segments")
    tab = pd.DataFrame(rows)
    q0 = tab[(tab.group == "regular_alt_batteries") & (tab.seg == 0)]["Q_Ah"].median()
    tab["SOH"] = tab["Q_Ah"] / q0
    tab.to_csv(OUT / "soh_table.csv", index=False)
    print("Q0 (Ah) =", q0)
    return tab
