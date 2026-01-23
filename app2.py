# app.py — Neighborhood Finder (LA-first + communities + Google hint + Multi-ZIP, light-blue style)
import os, re, json, io, zipfile, pathlib, requests, time, hashlib
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import folium
import geopandas as gpd
from shapely.geometry import Point, mapping
import pgeocode
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Neighborhood Finder", layout="wide")

# ---------------- Config / secrets ----------------
GOOGLE_API_KEY = os.getenv('API_KEY')

# Leaflet default-ish light blue polygon style
LIGHT_BLUE = "#3388ff"
FILL_OPACITY = 0.22
BORDER_WEIGHT = 2

# ---------------- Helpers ----------------
def arcgis_query_url(layer_url: str) -> str:
    layer_url = layer_url.rstrip("/")
    if layer_url.endswith(("/FeatureServer", "/MapServer")):
        layer_url = layer_url + "/0"
    return layer_url + "/query"

def arcgis_layer_to_geojson(layer_url: str, out_path: pathlib.Path, where="1=1", batch_size=2000):
    url = arcgis_query_url(layer_url)
    params = {"where": where, "outFields": "*", "returnGeometry": "true",
              "f": "geojson", "outSR": "4326", "resultRecordCount": batch_size, "resultOffset": 0}
    features = []; fc_template = None
    while True:
        r = requests.get(url, params=params, timeout=60); r.raise_for_status()
        chunk = r.json()
        if fc_template is None:
            fc_template = {k: v for k, v in chunk.items() if k != "features"}
        feats = chunk.get("features", [])
        features.extend(feats)
        if len(feats) < batch_size: break
        params["resultOffset"] += batch_size
    fc = {"type": "FeatureCollection", **(fc_template or {}), "features": features}
    out_path.write_text(json.dumps(fc))
    return out_path

def normalize_zip_any(s) -> str | None:
    if s is None: return None
    m = re.search(r'(\d{5})', str(s))
    return m.group(1) if m else None

# ---------------- Data bootstrap ----------------
DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(exist_ok=True)
PREV_FINDINGS_DIR = DATA_DIR / "previous_findings"
PREV_FINDINGS_DIR.mkdir(exist_ok=True)

# City of LA neighborhoods (Mapping L.A.)
LA_TIMES_FS = "https://services5.arcgis.com/7nsPwEMP38bSkCjy/arcgis/rest/services/LA_Times_Neighborhoods/FeatureServer/0"
la_times_path = DATA_DIR / "la_times_neighborhoods.geojson"
if not la_times_path.exists():
    arcgis_layer_to_geojson(LA_TIMES_FS, la_times_path)
la_nbhd = gpd.read_file(la_times_path).to_crs(4326)

# build a set of known mapping_la names for quick membership checks
_la_name_cols = ['name', 'NAME']
_la_names = set()
for c in _la_name_cols:
    if c in la_nbhd.columns:
        _la_names.update([str(x).strip() for x in la_nbhd[c].dropna().unique()])

# Census places (cities) – California
CA_PLACES_ZIP = DATA_DIR / "tl_2025_06_place.zip"
CA_PLACES_DIR = DATA_DIR / "tl_2025_06_place"
CA_PLACES_GPKG = DATA_DIR / "ca_places.gpkg"
if not CA_PLACES_GPKG.exists():
    if not CA_PLACES_ZIP.exists():
        url = "https://www2.census.gov/geo/tiger/TIGER2025/PLACE/tl_2025_06_place.zip"
        r = requests.get(url, timeout=120); r.raise_for_status()
        CA_PLACES_ZIP.write_bytes(r.content)
    if not CA_PLACES_DIR.exists():
        CA_PLACES_DIR.mkdir(exist_ok=True)
        with zipfile.ZipFile(CA_PLACES_ZIP, "r") as zf:
            zf.extractall(CA_PLACES_DIR)
    ca_places = gpd.read_file(str(CA_PLACES_DIR)).to_crs(4326)[["GEOID","NAME","geometry"]]
    ca_places.to_file(CA_PLACES_GPKG, layer="places", driver="GPKG")
la_cities = gpd.read_file(CA_PLACES_GPKG, layer="places").to_crs(4326)

# ---------------- Core polygon lookup ----------------
def find_feature(lat: float, lng: float):
    """Returns ('mapping_la'|'community'|'city', row) or (None, None)."""
    pt = gpd.GeoDataFrame(index=[0], geometry=[Point(lng, lat)], crs=4326)

    hit_nbhd = gpd.sjoin(pt, la_nbhd, how="left", predicate="within")
    if hit_nbhd.index_right.notna().any():
        row = la_nbhd.iloc[int(hit_nbhd.index_right.dropna().iloc[0])]
        return ("mapping_la", row)

    hit_city = gpd.sjoin(pt, la_cities, how="left", predicate="within")
    if hit_city.index_right.notna().any():
        row = la_cities.iloc[int(hit_city.index_right.dropna().iloc[0])]
        return ("city", row)

    return (None, None)

# ---------------- Geocoding helpers ----------------
zipgeo = pgeocode.Nominatim('us')

def latlng_from_zip(zipish):
    z5 = normalize_zip_any(zipish)
    if not z5: return (None, None)
    rec = zipgeo.query_postal_code(z5)
    if pd.isna(rec.latitude) or pd.isna(rec.longitude): return (None, None)
    return (float(rec.latitude), float(rec.longitude))

def address_to_latlng(address: str):
    if not GOOGLE_API_KEY: return (None, None)
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    r = requests.get(url, params={"address": address, "key": GOOGLE_API_KEY}, timeout=30)
    js = r.json()
    if js.get("status") == "OK" and js.get("results"):
        loc = js["results"][0]["geometry"]["location"]
        return float(loc["lat"]), float(loc["lng"])
    return (None, None)

def reverse_geocode_neighborhood(lat: float, lng: float):
    if not GOOGLE_API_KEY: return {}
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"latlng": f"{lat},{lng}", "key": GOOGLE_API_KEY, "result_type": "neighborhood|sublocality|political"}
    r = requests.get(url, params=params, timeout=30)
    js = r.json()
    if js.get("status") != "OK" or not js.get("results"): return {}
    best = {"neighborhood": None, "sublocality": None, "locality": None}
    for res in js["results"]:
        for comp in res.get("address_components", []):
            t = set(comp.get("types", [])); name = comp.get("long_name")
            if not best["neighborhood"] and ("neighborhood" in t or "sublocality_level_1" in t): best["neighborhood"] = name
            if not best["sublocality"] and ("sublocality" in t or "sublocality_level_1" in t): best["sublocality"] = name
            if not best["locality"] and ("locality" in t or "postal_town" in t): best["locality"] = name
    return best

# ---------------- Multi-ZIP helpers ----------------
def parse_zip_list(text: str) -> list[str]:
    if not text: return []
    raw = re.split(r'[,\n\r\t ]+', text.strip())
    zips = []
    for token in raw:
        m = re.search(r'(\d{5})', token)
        if m: zips.append(m.group(1))
    seen = set(); out = []
    for z in zips:
        if z not in seen:
            seen.add(z); out.append(z)
    return out

def lookup_zip_record(z: str, include_google_hint: bool):
    lat, lng = latlng_from_zip(z)
    if lat is None:
        return {"zip": z, "lat": None, "lng": None, "label_type": None, "label_name": None, "google_hint": None}
    layer, feat = find_feature(lat, lng)
    if layer == "mapping_la":
        label = feat.get("name") or feat.get("NAME") or "Unknown Neighborhood"
    elif layer == "community":
        label = feat.get("NAME") or feat.get("COMMUNITY") or feat.get("COMM_NAME") or "Unknown Community"
    elif layer == "city":
        label = feat.get("NAME", "Unknown City")
    else:
        label = None
    g_hint = None
    if include_google_hint and GOOGLE_API_KEY:
        g = reverse_geocode_neighborhood(lat, lng)
        g_hint = g.get("neighborhood") or g.get("sublocality") or g.get("locality")
    return {"zip": z, "lat": lat, "lng": lng, "label_type": layer, "label_name": label, "google_hint": g_hint}

# ---------------- Export helpers (LA County ZIP DB) ----------------
@st.cache_data(show_spinner=False)
def get_la_county_zip_rows():
    df = zipgeo._data.copy()
    df = df[(df["country_code"]=="US") & (df["state_code"]=="CA")
            & (df["county_name"].fillna("").str.lower()=="los angeles")
            & (df["postal_code"].str.len()==5)
            & df["latitude"].notna() & df["longitude"].notna()].copy()
    df = df[["postal_code","place_name","state_code","county_name","latitude","longitude"]]
    df = df.rename(columns={"postal_code":"zip","place_name":"place_name","state_code":"state",
                            "county_name":"county","latitude":"lat","longitude":"lng"})
    df["lat"] = df["lat"].astype(float); df["lng"] = df["lng"].astype(float)
    return df.reset_index(drop=True)

def best_label_for_point(lat, lng):
    layer, feat = find_feature(lat, lng)
    if layer == "mapping_la":
        label = feat.get("name") or feat.get("NAME") or "Unknown Neighborhood"
    elif layer == "community":
        label = feat.get("NAME") or feat.get("COMMUNITY") or feat.get("COMM_NAME") or "Unknown Community"
    elif layer == "city":
        label = feat.get("NAME", "Unknown City")
    else:
        label = None
    return layer, label

def build_la_zip_dataset(include_google_hint: bool):
    base = get_la_county_zip_rows()
    out_rows = []
    prog = st.progress(0, text="Building LA County ZIP database…")
    total = len(base)
    for i, row in base.iterrows():
        lat = float(row["lat"]); lng = float(row["lng"])
        layer, label = best_label_for_point(lat, lng)
        g_hint = None
        if include_google_hint and GOOGLE_API_KEY:
            g = reverse_geocode_neighborhood(lat, lng)
            g_hint = g.get("neighborhood") or g.get("sublocality") or g.get("locality")
        out_rows.append({"zip": row["zip"], "lat": lat, "lng": lng,
                         "label_type": layer, "label_name": label, "google_hint": g_hint})
        if (i+1) % 5 == 0 or i == total-1:
            prog.progress((i+1)/total)
    prog.empty()
    return pd.DataFrame(out_rows).sort_values("zip").reset_index(drop=True)

# ---------------- Previous findings storage helpers ----------------
def is_valid_prev_value(v: str) -> bool:
    if v is None: return False
    s = str(v).strip()
    if s == "": return False
    s_low = s.lower()
    if s_low in {"#ref!","nan","none","null","n/a","na"}: return False
    return True

def save_prev_finding_file(uploaded_file):
    """
    Save uploaded findings CSV into PREV_FINDINGS_DIR with hashed name to avoid collisions.
    Only keep rows that have a valid 'recommended' value and contact_zip_code; deduplicate
    against existing stored rows by (zip, zip4) so we only store *new* mappings.
    """
    # read uploaded content into dataframe
    try:
        content = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        if isinstance(content, (bytes, bytearray)):
            csv_text = content.decode("utf-8", errors="replace")
        else:
            csv_text = str(content)
        df_new = pd.read_csv(StringIO(csv_text), dtype=str).fillna("")
    except Exception as e:
        raise RuntimeError(f"Unable to parse CSV: {e}")

    # normalize column names
    df_new.columns = [c.strip() for c in df_new.columns]

    # ensure required columns exist or attempt to guess
    # We'll look for contact_zip_code, contact_zip4_code, recommended (case-insensitive)
    cols_lower = {c.lower(): c for c in df_new.columns}
    if "contact_zip_code" not in cols_lower:
        # try common alternatives
        for alt in ["zip","zip_code","postal_code","contact_zip"]:
            if alt in cols_lower:
                cols_lower["contact_zip_code"] = cols_lower[alt]
                break
    if "contact_zip4_code" not in cols_lower:
        for alt in ["zip4","zip_4","postal_code_4","contact_zip4"]:
            if alt in cols_lower:
                cols_lower["contact_zip4_code"] = cols_lower[alt]
                break
    if "recommended" not in cols_lower:
        for alt in ["recommended_neighborhood","neighborhood","recommended_location","recommended_place"]:
            if alt in cols_lower:
                cols_lower["recommended"] = cols_lower[alt]
                break

    # create a working small dataframe with normalized column names
    df_small = pd.DataFrame()
    df_small["contact_zip_code"] = df_new[cols_lower.get("contact_zip_code", "")] if cols_lower.get("contact_zip_code") in df_new.columns else ""
    df_small["contact_zip4_code"] = df_new[cols_lower.get("contact_zip4_code", "")] if cols_lower.get("contact_zip4_code") in df_new.columns else ""
    df_small["recommended"] = df_new[cols_lower.get("recommended", "")] if cols_lower.get("recommended") in df_new.columns else ""

    # normalize zip and zip4
    df_small["contact_zip_code"] = df_small["contact_zip_code"].astype(str).str.strip().str[:5]
    df_small["contact_zip4_code"] = df_small["contact_zip4_code"].astype(str).str.strip()

    # keep only rows with valid recommended value
    df_small = df_small[df_small["recommended"].apply(is_valid_prev_value)].copy()
    if df_small.empty:
        # nothing to save
        return None

    # load existing previous findings and dedupe: keep only rows not already present
    existing = load_all_prev_findings()
    existing_keys = set()
    for _, r in existing.iterrows():
        k = (str(r.get("contact_zip_code","")).strip()[:5], str(r.get("contact_zip4_code","")).strip())
        existing_keys.add(k)

    # choose rows that are new keys
    rows_to_save = []
    for _, r in df_small.iterrows():
        k = (str(r.get("contact_zip_code","")).strip()[:5], str(r.get("contact_zip4_code","")).strip())
        if k not in existing_keys:
            rows_to_save.append({"contact_zip_code": k[0], "contact_zip4_code": k[1], "recommended": str(r["recommended"]).strip()})
            existing_keys.add(k)

    if not rows_to_save:
        # nothing new to save
        return None

    out_df = pd.DataFrame(rows_to_save)

    # write out file with timestamp + hash (content small)
    content_bytes = out_df.to_csv(index=False).encode("utf-8")
    h = hashlib.sha1(content_bytes).hexdigest()[:8]
    ts = int(time.time())
    fname = f"findings_{ts}_{h}.csv"
    path = PREV_FINDINGS_DIR / fname
    path.write_bytes(content_bytes)
    return fname

def list_prev_finding_files():
    return sorted([p.name for p in PREV_FINDINGS_DIR.glob("*.csv")])

def load_all_prev_findings() -> pd.DataFrame:
    files = list_prev_finding_files()
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(PREV_FINDINGS_DIR / f, dtype=str).fillna("")
            # normalize column names lowercase for safety
            df.columns = [c.strip() for c in df.columns]
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(columns=["contact_zip_code","contact_zip4_code","recommended"])
    out = pd.concat(dfs, ignore_index=True, sort=False).astype(str).fillna("")
    # ensure these columns exist at least
    for c in ["contact_zip_code","contact_zip4_code","recommended"]:
        if c not in out.columns:
            out[c] = ""
    # normalize zip formatting (zip5 and optional zip4)
    out["contact_zip_code"] = out["contact_zip_code"].astype(str).str.strip().str[:5]
    out["contact_zip4_code"] = out["contact_zip4_code"].astype(str).str.replace(r'\.0$','', regex=True).str.strip()
    # dedupe keeping first occurrence
    out = out.drop_duplicates(subset=["contact_zip_code","contact_zip4_code"], keep="first").reset_index(drop=True)
    return out

def delete_prev_finding_file(filename):
    p = PREV_FINDINGS_DIR / filename
    if p.exists(): p.unlink()
    return

# ---------------- UI ----------------
st.title("Neighborhood Finder (LA-first)")

tab_search, tab_findings, tab_batch, tab_dedupe = st.tabs(
    ["Search & ZIP List", "Findings Manager", "Batch Neighborhood", "Dedupe & Export"]
)

# ---- Search & ZIP List (combined Single + ZIP List) ----
# ---- Combined Search Tab (replaces previous "Single Search" + "ZIP List") ----

with tab_search:
    # top-level choice: single vs multi
    mode_choice = st.radio("Mode:", ["Single Lookup", "ZIP List (multi)"], index=0, horizontal=True)

    if mode_choice == "Single Lookup":
        c1, c2 = st.columns([1, 3], gap="large")
        with c1:
            search_by = st.radio("Search by:", ["ZIP Code", "Address", "Lat/Lng"], index=0)
            lat = lng = None
            if search_by == "ZIP Code":
                z = st.text_input("ZIP Code (5 or ZIP+4)", value="90026", key="single_zip")
                if z:
                    lat, lng = latlng_from_zip(z)
                    if lat is None:
                        st.warning("Enter a valid 5-digit ZIP or ZIP+4 (e.g., 90004-3878).")
            elif search_by == "Address":
                addr = st.text_input("Address", placeholder="1632 Bellevue Ave, Los Angeles, CA 90026", key="single_addr")
                if addr:
                    lat, lng = address_to_latlng(addr)
                    if lat is None:
                        st.info("Set GOOGLE_API_KEY to enable address geocoding.")
            else:
                lat = st.number_input("Latitude", value=34.0901, format="%.7f", key="single_lat")
                lng = st.number_input("Longitude", value=-118.2606, format="%.7f", key="single_lng")

            show_google_hint = st.checkbox("Show Google 'neighborhood' hint", value=True, key="single_google_hint")
            zoom = st.slider("Map zoom", 8, 18, 13, key="single_zoom")

        with c2:
            m = folium.Map(location=[34.05, -118.24], zoom_start=zoom, control_scale=True, tiles="OpenStreetMap")
            info_box = st.empty()

            if lat is not None and lng is not None:
                layer, feat = find_feature(lat, lng)
                folium.Marker([lat, lng], tooltip="Query point").add_to(m)

                if layer and feat is not None and feat.geometry is not None:
                    gj = folium.GeoJson(
                        data=mapping(feat.geometry),
                        name="Match",
                        style_function=lambda _s: {"color": LIGHT_BLUE, "fillColor": LIGHT_BLUE,
                                                   "fillOpacity": FILL_OPACITY, "weight": BORDER_WEIGHT},
                        tooltip=f"{'Neighborhood' if layer=='mapping_la' else ('Community' if layer=='community' else 'City')}: "
                                f"{feat.get('name', feat.get('NAME', feat.get('COMMUNITY','Unknown')))}"
                    ); gj.add_to(m)
                    try: m.fit_bounds(gj.get_bounds(), padding=(30, 30))
                    except Exception: pass

                lines = []
                if layer == "mapping_la":
                    name = feat.get("name") or feat.get("NAME") or "Unknown Neighborhood"
                    lines.append(f"**Neighborhood (Mapping L.A.):** {name}")
                elif layer == "community":
                    nm = feat.get("NAME") or feat.get("COMMUNITY") or feat.get("COMM_NAME") or "Unknown Community"
                    lines.append(f"**Community (County):** {nm}")
                elif layer == "city":
                    city_name = feat.get("NAME", "Unknown City")
                    lines.append(f"**City (Census ‘place’):** {city_name}")
                else:
                    lines.append("**No neighborhood/city match found.**")

                if show_google_hint and GOOGLE_API_KEY:
                    g = reverse_geocode_neighborhood(lat, lng)
                    g_label = g.get("neighborhood") or g.get("sublocality") or g.get("locality")
                    lines.append(f"*Google neighborhood hint:* {g_label or '(none)'}")

                (info_box.success if layer=="mapping_la" else info_box.info if layer in {"community","city"} else info_box.warning)(
                    "  \n".join(lines)
                )

            folium.LayerControl(collapsed=False).add_to(m)
            st_folium(m, use_container_width=True)

    else:  # ZIP List (multi)
        c1, c2 = st.columns([1, 3], gap="large")
        with c1:
            zlist_text = st.text_area(
                "Enter ZIPs or ZIP+4 (comma/newline separated)",
                value="90012, 90007, 90018",
                height=140,
                key="multi_text"
            )
            include_google_multi = st.checkbox("Include Google hint for each ZIP (uses API quota)", value=False, key="multi_google")
            show_markers = st.checkbox("Show centroid markers", value=False, key="multi_markers")
            run_multi = st.button("Lookup ZIPs", type="primary", key="multi_run")

        with c2:
            m2 = folium.Map(location=[34.05, -118.24], zoom_start=11, control_scale=True, tiles="OpenStreetMap")

            if run_multi:
                zips = parse_zip_list(zlist_text)
                if not zips:
                    st.warning("No valid ZIPs found. Add 5-digit ZIPs or ZIP+4.")
                else:
                    with st.spinner(f"Looking up {len(zips)} ZIPs…"):
                        rows = [lookup_zip_record(z, include_google_multi) for z in zips]
                        df_multi = pd.DataFrame(rows)
                    st.session_state["df_multi"] = df_multi

            df_multi = st.session_state.get("df_multi")
            if isinstance(df_multi, pd.DataFrame) and not df_multi.empty:
                # Download + table
                buf = StringIO(); df_multi.to_csv(buf, index=False)
                st.download_button("Download ZIP results CSV",
                                   data=buf.getvalue().encode("utf-8"),
                                   file_name="zip_results.csv", mime="text/csv",
                                   key="ziplist_download_combined")
                st.dataframe(df_multi, use_container_width=True, height=260)

                # Draw polygons
                fg_polys = folium.FeatureGroup(name="Neighborhood polygons", show=True)
                all_bounds = []
                for _, r in df_multi.dropna(subset=["lat", "lng"]).iterrows():
                    layer, feat = find_feature(float(r["lat"]), float(r["lng"]))
                    if not (layer and feat is not None and feat.geometry is not None):
                        continue
                    nm = feat.get("name", feat.get("NAME", feat.get("COMMUNITY", "Unknown")))
                    gj = folium.GeoJson(
                        mapping(feat.geometry),
                        name=f"{r['zip']} — {nm}",
                        style_function=lambda _s: {"color": LIGHT_BLUE, "fillColor": LIGHT_BLUE,
                                                   "fillOpacity": FILL_OPACITY, "weight": BORDER_WEIGHT},
                        tooltip=f"{r['zip']} — {('Neighborhood' if layer=='mapping_la' else ('Community' if layer=='community' else 'City'))}: {nm}",
                    )
                    gj.add_to(fg_polys)
                    try:
                        b = gj.get_bounds(); all_bounds.extend(b)
                    except Exception:
                        pass
                fg_polys.add_to(m2)

                # Optional centroid markers
                if show_markers:
                    pts = df_multi[["lat","lng","zip","label_name","label_type","google_hint"]].dropna()
                    fg_pts = folium.FeatureGroup(name="ZIP markers", show=True)
                    for _, r in pts.iterrows():
                        pop = (
                            f"<b>ZIP {r['zip']}</b><br>"
                            f"{(r['label_type'] or '').title()}: {r['label_name'] or ''}<br>"
                            f"{'Google: ' + str(r['google_hint']) if pd.notna(r['google_hint']) and r['google_hint'] else ''}"
                        )
                        folium.CircleMarker([r["lat"], r["lng"]], radius=5, tooltip=pop, popup=pop).add_to(fg_pts)
                    fg_pts.add_to(m2)

                # Focus ZIP (emphasize outline)
                options = list(df_multi["zip"].astype(str))
                focus_zip = st.selectbox("Focus ZIP (emphasize outline):", options, index=0, key="focus_zip_combined")
                f_lat, f_lng = latlng_from_zip(focus_zip)
                if f_lat is not None:
                    layer_f, feat_f = find_feature(f_lat, f_lng)
                    if layer_f and feat_f is not None and feat_f.geometry is not None:
                        nm_f = feat_f.get("name", feat_f.get("NAME", feat_f.get("COMMUNITY", "Unknown")))
                        folium.GeoJson(
                            mapping(feat_f.geometry),
                            name=f"Selected {focus_zip}",
                            style_function=lambda _s: {"color": "#111", "fillOpacity": 0, "weight": 4},
                            tooltip=f"{('Neighborhood' if layer_f=='mapping_la' else ('Community' if layer_f=='community' else 'City'))}: {nm_f}",
                        ).add_to(m2)

                if all_bounds:
                    try: m2.fit_bounds(all_bounds, padding=(20, 20))
                    except Exception: pass

            folium.LayerControl(collapsed=False).add_to(m2)
            st_folium(m2, use_container_width=True)

        # CSV upload + enrichment (same behavior as previous ZIP List)
        st.markdown("### 📂 Upload CSV for ZIP + ZIP4 Lookup")
        uploaded_csv = st.file_uploader("Upload CSV with 'contact_zip_code' and optional 'contact_zip4_code' columns", type=["csv"], key="multi_upload")

        if uploaded_csv is not None:
            df_in = pd.read_csv(uploaded_csv, dtype=str).fillna("")

            if "contact_zip_code" not in df_in.columns:
                st.error("CSV must have a column named 'contact_zip_code'.")
            else:
                rows_out = []
                for _, row in df_in.iterrows():
                    z = str(row["contact_zip_code"]).strip()

                    if "contact_zip4_code" in df_in.columns and pd.notna(row["contact_zip4_code"]):
                        z_full = f"{z[:5]}-{str(row['contact_zip4_code']).zfill(4)}"
                    else:
                        z_full = z

                    lat, lng = latlng_from_zip(z_full)
                    layer, feat = find_feature(lat, lng) if lat is not None else (None, None)

                    neighborhood = None
                    source = None
                    if layer and feat is not None:
                        if layer == "mapping_la":
                            neighborhood = feat.get("name") or feat.get("NAME")
                            source = "mapping_la"
                        elif layer == "community":
                            neighborhood = feat.get("NAME") or feat.get("COMMUNITY") or feat.get("COMM_NAME")
                            source = "community"
                        elif layer == "city":
                            neighborhood = feat.get("NAME")
                            source = "city"
                    else:
                        g = reverse_geocode_neighborhood(lat, lng) if (lat and lng) else {}
                        neighborhood = g.get("neighborhood") or g.get("sublocality") or g.get("locality")
                        source = "google_hint" if neighborhood else None

                    rows_out.append({
                        **row.to_dict(),
                        "neighborhood_suggested": neighborhood,
                        "source": source
                    })

                df_out = pd.DataFrame(rows_out)

                st.dataframe(df_out, use_container_width=True, height=300)

                out_buf = StringIO(); df_out.to_csv(out_buf, index=False)
                st.download_button("⬇️ Download Enriched CSV",
                                   data=out_buf.getvalue().encode("utf-8"),
                                   file_name="neighborhood_enriched.csv",
                                   mime="text/csv",
                                   key="multi_enriched_download")


# ---- Findings Manager ----
with tab_findings:
    st.header("Findings Manager — upload & manage previous neighborhood findings")
    st.write("Upload one or more CSVs that contain previous neighborhood findings. These files should have at least: `contact_zip_code`, `contact_zip4_code`, `recommended` (neighborhood).")
    uploaded = st.file_uploader("Upload previous findings CSV(s)", accept_multiple_files=True, type=["csv"])
    if uploaded:
        saved = []
        new_count = 0
        for up in uploaded:
            try:
                fname = save_prev_finding_file(up)
                if fname:
                    saved.append(fname)
                    new_count += 1
            except Exception as e:
                st.error(f"Failed to save {up.name}: {e}")
        if saved:
            st.success(f"Saved {len(saved)} file(s) with new rows ({new_count} new zip mappings total).")
        else:
            st.info("No new valid rows found to save (either no recommended values or all were already stored).")

    st.markdown("#### Existing stored findings files")
    files = list_prev_finding_files()
    if not files:
        st.info("No previous findings uploaded yet.")
    else:
        for f in files:
            cols = st.columns([0.7, 0.15, 0.15])
            cols[0].write(f)
            if cols[1].button("Preview", key=f"prev_{f}"):
                try:
                    dfp = pd.read_csv(PREV_FINDINGS_DIR / f, dtype=str).fillna("").head(50)
                    st.dataframe(dfp)
                except Exception as e:
                    st.error(f"Unable to preview {f}: {e}")
            # Simplified delete: single confirm button
            if cols[2].button("Delete", key=f"delbtn_{f}"):
                st.session_state["to_delete"] = f
        if st.session_state.get("to_delete"):
            target = st.session_state["to_delete"]
            try:
                delete_prev_finding_file(target)
                st.success(f"Deleted {target}")
                st.session_state["to_delete"] = None
                    # refresh files
                files = list_prev_finding_files()
            except Exception as e:
                st.error(f"Delete failed: {e}")

    # allow quick load/inspect of all combined findings
    if st.button("Load & preview combined findings (first 200 rows)"):
        df_comb = load_all_prev_findings()
        st.write(f"Combined rows: {len(df_comb)}")
        st.dataframe(df_comb.head(200))

# ---- Batch Neighborhood (core multi-row processing) ----
# ---- Batch tab (with progress bar + reload/reset without full-page refresh) ----
from datetime import datetime
import json, time

with tab_batch:
    st.header("Batch Neighborhood Matching")

    # paths for persistence
    DATA_DIR.mkdir(exist_ok=True)
    full_path = DATA_DIR / "batch_full_results.csv"
    unmatched_path = DATA_DIR / "batch_unmatched_results.csv"
    meta_path = DATA_DIR / "batch_meta.json"

    # ensure session_state keys exist (won't be cleared on download)
    if "batch_full" not in st.session_state:
        st.session_state["batch_full"] = None
    if "batch_unmatched" not in st.session_state:
        st.session_state["batch_unmatched"] = None
    if "batch_meta" not in st.session_state:
        st.session_state["batch_meta"] = {}

    # try to lazy-load persisted files once (keeps them until manual refresh or explicit reload)
    if st.session_state["batch_full"] is None and full_path.exists():
        try:
            st.session_state["batch_full"] = pd.read_csv(full_path, dtype=str).fillna("")
        except Exception:
            st.warning("Could not load persisted full results from disk (will proceed without).")

    if st.session_state["batch_unmatched"] is None and unmatched_path.exists():
        try:
            st.session_state["batch_unmatched"] = pd.read_csv(unmatched_path, dtype=str).fillna("")
        except Exception:
            st.warning("Could not load persisted unmatched results from disk (will proceed without).")

    if not st.session_state["batch_meta"] and meta_path.exists():
        try:
            st.session_state["batch_meta"] = json.loads(meta_path.read_text())
        except Exception:
            st.session_state["batch_meta"] = {}

    # UI controls
    uploaded_file = st.file_uploader("Upload CSV for neighborhood enrichment", type=["csv"])
    run_batch = st.button("Run Batch Neighborhood Matching", type="primary", key="run_batch")
    clear_persist = st.button("Reset", key="clear_persist")

    # clear persisted (explicit)
    if clear_persist:
        for k in ["batch_full", "batch_unmatched", "batch_meta"]:
            if k in st.session_state:
                st.session_state.pop(k)
        for p in (full_path, unmatched_path, meta_path):
            try:
                if p.exists(): p.unlink()
            except Exception:
                pass
        st.success("Session resetted")

    

    # show any persisted run info
    if st.session_state.get("batch_full") is not None or st.session_state.get("batch_unmatched") is not None:
        meta = st.session_state.get("batch_meta", {})
        st.markdown("### ✅ Persisted results (loaded into session)")
        if meta.get("timestamp"):
            st.write(f"**Last run:** {meta.get('timestamp')}  —  **Duration:** {meta.get('duration_s', 0):.2f}s")
        if st.session_state.get("batch_full") is not None:
            st.markdown("**Full results preview**")
            st.dataframe(st.session_state["batch_full"].head(20), use_container_width=True)
            buf = StringIO(); st.session_state["batch_full"].to_csv(buf, index=False)
            st.download_button("Download persisted full results CSV",
                               data=buf.getvalue().encode("utf-8"),
                               file_name="neighborhoods_full_no_recommendation.csv",
                               mime="text/csv",
                               key="download_persisted_full")
        if st.session_state.get("batch_unmatched") is not None:
            st.markdown("**Unmatched rows preview**")
            st.dataframe(st.session_state["batch_unmatched"].head(50), use_container_width=True)
            buf2 = StringIO(); st.session_state["batch_unmatched"].to_csv(buf2, index=False)
            st.download_button("Download persisted unmatched CSV",
                               data=buf2.getvalue().encode("utf-8"),
                               file_name="unmatched_after_matching.csv",
                               mime="text/csv",
                               key="download_persisted_unmatched")

    # --- Run the batch when requested ---
    if run_batch:
        if not uploaded_file:
            st.warning("Please upload a CSV before running the batch.")
        else:
            start_time = time.perf_counter()
            with st.spinner("Processing batch neighborhood matching..."):
                df = pd.read_csv(uploaded_file, dtype=str).fillna("")
                prev = load_all_prev_findings()

                # Prepare progress UI
                total_steps = 3  # step1, step2, step3 loops
                prog_container = st.container()
                prog = prog_container.progress(0.0)
                status_text = prog_container.empty()

                # --- STEP 1: Match by ZIP + ZIP4 using previous findings ---
                status_text.info("Step 1/3 — Matching by ZIP/ZIP4 against previous findings...")
                rows = len(df)
                if rows == 0:
                    prog.progress(1.0)
                else:
                    df["neighborhood"] = None
                    for i, row in enumerate(df.itertuples(index=False), start=1):
                        # access via row.__getattribute__ or index-based; keep using original approach for clarity
                        z5 = str(df.at[i-1, "contact_zip_code"] if "contact_zip_code" in df.columns else "").strip()[:5]
                        z4 = str(df.at[i-1, "contact_zip4_code"] if "contact_zip4_code" in df.columns else "").strip()
                        match = prev[
                            (prev["contact_zip_code"] == z5) &
                            (prev["contact_zip4_code"] == z4)
                        ]
                        if not match.empty:
                            df.at[i-1, "neighborhood"] = match.iloc[0]["recommended"]
                        # update progress fraction for step1 (0..0.33)
                        prog.progress((i/rows) * (1/total_steps))

                # --- STEP 2: Fill missing neighborhoods via LA Times ---
                missing_idx = df[df["neighborhood"].isna()].index.tolist()
                status_text.info("Step 2/3 — Filling missing neighborhoods via Mapping L.A. polygons...")
                mcount = len(missing_idx)
                if mcount == 0:
                    prog.progress(2/3)
                else:
                    for j, i in enumerate(missing_idx, start=1):
                        z5 = normalize_zip_any(df.at[i, "contact_zip_code"])
                        lat, lng = latlng_from_zip(z5)
                        if lat and lng:
                            layer, feat = find_feature(lat, lng)
                            if layer == "mapping_la":
                                df.at[i, "neighborhood"] = feat.get("name") or feat.get("NAME")
                        prog.progress((1/total_steps) + (j/mcount) * (1/total_steps))

                # --- STEP 3: For remaining nulls, use Google/TIGER fallback ---
                missing_idx2 = df[df["neighborhood"].isna()].index.tolist()
                status_text.info("Step 3/3 — Google/TIGER fallback for remaining rows...")
                m2 = len(missing_idx2)
                df["neighborhood_recommended"] = None
                if m2 == 0:
                    prog.progress(1.0)
                else:
                    for k, i in enumerate(missing_idx2, start=1):
                        z5 = normalize_zip_any(df.at[i, "contact_zip_code"])
                        lat, lng = latlng_from_zip(z5)
                        if lat and lng:
                            rec = {}
                            if GOOGLE_API_KEY:
                                rec = reverse_geocode_neighborhood(lat, lng)
                            else:
                                pt = gpd.GeoDataFrame(index=[0], geometry=[Point(lng, lat)], crs=4326)
                                hit_city = gpd.sjoin(pt, la_cities, how="left", predicate="within")
                                if hit_city.index_right.notna().any():
                                    rec["locality"] = la_cities.iloc[int(hit_city.index_right.dropna().iloc[0])]["NAME"]
                            df.at[i, "neighborhood_recommended"] = (
                                rec.get("neighborhood") or rec.get("sublocality") or rec.get("locality")
                            )
                        prog.progress((2/total_steps) + (k/m2) * (1/total_steps))

                # --- Post-processing (STEP 4 etc) ---
                status_text.info("Finalizing results...")
                for i, row in df.iterrows():
                    nrec = row.get("neighborhood_recommended", "")
                    ccity = row.get("contact_city", "")
                    if not pd.isna(nrec) and str(nrec).strip().lower() == str(ccity).strip().lower():
                        df.at[i, "neighborhood"] = nrec

                # finalize masks / outputs
                def is_neighborhood_null(val):
                    return pd.isna(val) or str(val).strip() == ""

                null_mask = df["neighborhood"].apply(is_neighborhood_null)
                if "neighborhood_recommended" in df.columns:
                    df_full_display = df.drop(columns=["neighborhood_recommended"]).copy()
                else:
                    df_full_display = df.copy()
                df_full_display = df_full_display[~df_full_display["neighborhood"].isna() & (df_full_display["neighborhood"].str.strip() != "")]

                if "neighborhood_recommended" not in df.columns:
                    df["neighborhood_recommended"] = None

                unmatched_after_all = df[null_mask].copy()
                cols_needed = ["contact_zip_code", "contact_zip4_code", "contact_city", "neighborhood", "neighborhood_recommended"]
                for c in cols_needed:
                    if c not in unmatched_after_all.columns:
                        unmatched_after_all[c] = None
                unmatched_export = unmatched_after_all[cols_needed].reset_index(drop=True)

                # persist to session_state and disk (so they remain until manual refresh / explicit clear)
                st.session_state["batch_full"] = df_full_display.fillna("").astype(str)
                st.session_state["batch_unmatched"] = unmatched_export.fillna("").astype(str)

                try:
                    st.session_state["batch_full"].to_csv(full_path, index=False)
                    st.session_state["batch_unmatched"].to_csv(unmatched_path, index=False)
                except Exception as e:
                    st.warning(f"Could not persist CSVs to disk: {e}")

                duration = time.perf_counter() - start_time
                meta = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "duration_s": duration, "rows_input": len(df)}
                st.session_state["batch_meta"] = meta
                try:
                    meta_path.write_text(json.dumps(meta))
                except Exception:
                    pass

                # finalize progress UI and show results
                prog.progress(1.0)
                status_text.success("Batch processing complete.")

                st.success("✅ Neighborhood matching completed.")
                st.markdown(f"**Run time:** {duration:.2f} seconds — **Rows processed:** {len(df)}")
                st.markdown("### Full results (without `neighborhood_recommended`) — preview")
                st.dataframe(st.session_state["batch_full"].head(20), use_container_width=True)

                # Download buttons (do not clear session_state)
                buf_full = StringIO()
                st.session_state["batch_full"].to_csv(buf_full, index=False)
                st.download_button(
                    "Download full results (no neighborhood_recommended)",
                    data=buf_full.getvalue().encode("utf-8"),
                    file_name="neighborhoods_full_no_recommendation.csv",
                    mime="text/csv",
                    key="download_full_now"
                )

                if not st.session_state["batch_unmatched"].empty:
                    st.success(f"Saved {len(st.session_state['batch_unmatched'])} unmatched rows (persisted).")
                    st.markdown("### Unmatched rows (neighborhood still null) — preview")
                    st.dataframe(st.session_state["batch_unmatched"].head(50), use_container_width=True)

                    buf_un = StringIO()
                    st.session_state["batch_unmatched"].to_csv(buf_un, index=False)
                    st.download_button(
                        "Download unmatched rows (neighborhood still null)",
                        data=buf_un.getvalue().encode("utf-8"),
                        file_name="unmatched_after_matching.csv",
                        mime="text/csv",
                        key="download_unmatched_now"
                    )
                else:
                    st.info("No rows remain unmatched (neighborhood null) after processing.")




# ---- Dedupe & Export ----
with tab_dedupe:
    st.header("Dedupe & Select Columns")
    st.write("Upload CSV to dedupe on a chosen column, then select which columns to keep and optionally rename them.")
    uploaded_dedupe = st.file_uploader("CSV to dedupe", type=["csv"], key="dedupe_upload")
    if uploaded_dedupe is not None:
        df_d = pd.read_csv(uploaded_dedupe, dtype=str).fillna("")
        st.write(f"Columns detected ({len(df_d.columns)}):")
        cols = list(df_d.columns)
        st.write(cols)
        dedupe_col = st.selectbox("Choose the column to dedupe on (keep first occurrence)", options=cols)
        if dedupe_col:
            # count duplicates (including blanks)
            # we'll count rows with empty dedupe_col as non-dedupable and keep them
            total_before = len(df_d)
            # rows considered for dedupe: non-empty keys
            keys = df_d[dedupe_col].astype(str).str.strip()
            non_empty = keys != ""
            duplicates_count = 0
            if non_empty.any():
                df_nonempty = df_d[non_empty].copy()
                # compute duplicates based on dedupe_col
                duplicates_count = len(df_nonempty) - len(df_nonempty.drop_duplicates(subset=[dedupe_col], keep="first"))
            # dedupe: drop duplicates keeping first occurrence for all rows
            df_final = df_d.drop_duplicates(subset=[dedupe_col], keep="first")
            st.write(f"Duplicates found & removed (by `{dedupe_col}`): {duplicates_count}")
            st.write(f"Rows before: {total_before}, after dedupe: {len(df_final)} (keeps first occurrence).")

            # column selection
            st.markdown("Select columns to include in final output:")
            selected = st.multiselect("Columns to keep", options=list(df_final.columns), default=list(df_final.columns))
            rename_map = {}
            if selected:
                st.markdown("Optionally provide new names for selected columns:")
                for c in selected:
                    new = st.text_input(f"Rename `{c}` to:", value=c, key=f"rename_{c}")
                    rename_map[c] = new.strip()
                df_out = df_final[selected].copy()
                df_out.rename(columns=rename_map, inplace=True)
                st.dataframe(df_out.head(20), use_container_width=True)
                buf = StringIO(); df_out.to_csv(buf, index=False)
                st.download_button("Download deduped & selected CSV", data=buf.getvalue().encode("utf-8"),
                                   file_name="deduped_selected.csv", mime="text/csv")


