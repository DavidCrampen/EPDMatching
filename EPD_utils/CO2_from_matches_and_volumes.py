import numpy as np
import ifcopenshell
import ifcopenshell.geom
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import ConvexHull
import os

# Map common variations of EPD units to a canonical form
UNIT_MAP = {
    "m2": "m2", "m²": "m2", "qm": "m2",
    "m3": "m3",
    "kg": "kg"
}

# Default configuration for your current Ökobau dataset
DEFAULT_EPD_CONFIG = {
    "uuid_col":       "UUID",
    "ref_unit_col":   "Bezugseinheit",            # reference unit column (e.g. m2 / m3 / kg)
    "ref_size_col":   "Bezugsgroesse",           # reference size column (e.g. 1 m2, 1 m3)
    "co2_cols":       ["GWPtotal (A2)", "GWP"],  # prioritized CO2 columns (first valid one is used)
    "density_col":    "Rohdichte (kg/m3)",       # density column when reference is kg
    "epd_name_col":   "Name (de)",               # optional, only for debug/print
}

def compute_mesh_volume(mesh):
    """
    Compute a watertight mesh volume by summing tetrahedra
    about the mesh centroid (rather than the global origin).
    """
    vertices = np.array(mesh.verts).reshape(-1, 3)
    faces = np.array(mesh.faces).reshape(-1, 3)
    centroid = vertices.mean(axis=0)
    vol = 0.0
    for tri in faces:
        a, b, c = vertices[tri] - centroid
        vol += np.dot(a, np.cross(b, c)) / 6.0
    return abs(vol)

def area_by_ransac_plane(verts):
    """
    Fit a plane with RANSAC, then project verts onto two
    orthonormal axes in that plane and compute convex-hull area.
    """
    # fit z = a x + b y
    X = verts[:, :2]
    y = verts[:, 2]
    ransac = RANSACRegressor()
    ransac.fit(X, y)
    a, b = ransac.estimator_.coef_
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)

    # build in-plane axes u,v
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # project and compute convex-hull area
    proj2d = np.vstack((verts.dot(u), verts.dot(v))).T
    try:
        hull = ConvexHull(proj2d)
        return hull.volume  # 2D hull "volume" equals area
    except Exception:
        # Fallback to bounding rectangle if hull fails
        mins = proj2d.min(axis=0)
        maxs = proj2d.max(axis=0)
        return np.prod(maxs - mins)

def get_material_layers(element):
    """
    Return material layers and the usage object if available.

    Returns:
        (layers, usage) where:
          - layers: list of IfcMaterialLayer
          - usage:  IfcMaterialLayerSetUsage or None
    """
    for rel in getattr(element, "HasAssociations", []):
        if rel.is_a("IfcRelAssociatesMaterial"):
            mat = rel.RelatingMaterial
            if mat.is_a("IfcMaterialLayerSetUsage"):
                return mat.ForLayerSet.MaterialLayers, mat
            elif mat.is_a("IfcMaterialLayerSet"):
                return mat.MaterialLayers, None
    return None, None

def get_epd_row_by_uuid(epd_df, uuid, epd_config):
    """
    Return a single EPD row matching the UUID using the configured UUID column.
    """
    uuid_col = epd_config["uuid_col"]
    if uuid_col not in epd_df.columns:
        raise KeyError(f"UUID column '{uuid_col}' not found in EPD dataframe.")
    row = epd_df[epd_df[uuid_col] == uuid]
    return row.iloc[0] if not row.empty else None

def _pick_first_valid_co2(epd_row, co2_cols):
    """
    Select the first CO2 column in co2_cols that exists and is numeric.
    """
    for col in co2_cols:
        if col in epd_row.index:
            val = pd.to_numeric(epd_row[col], errors="coerce")
            if not pd.isna(val):
                return val
    return np.nan

def calculate_quantity_and_co2(el, entry, mesh, epd_row, epd_config, layer=None, thickness=None):
    """
    Compute amount, volume, area, and total CO2 eq based on an EPD configuration.

    Args:
        el:         IFC element
        entry:      matched EPD entry (with type, layer_index, etc.)
        mesh:       ifcopenshell mesh geometry
        epd_row:    pandas.Series row from EPD dataframe
        epd_config: dict defining EPD column names
        layer:      IfcMaterialLayer or None
        thickness:  layer thickness from IFC or None

    Returns:
        dict with keys:
            co2_total, amount, co2_per_unit, density, reference, volume, area
    """

    ref_unit_col = epd_config["ref_unit_col"]
    ref_size_col = epd_config["ref_size_col"]
    co2_cols     = epd_config["co2_cols"]
    density_col  = epd_config["density_col"]
    epd_name_col = epd_config.get("epd_name_col")

    # Reference unit
    if ref_unit_col not in epd_row.index:
        raise KeyError(f"Reference unit column '{ref_unit_col}' not found in EPD row.")

    raw_ref = str(epd_row[ref_unit_col]).strip().lower()
    reference = UNIT_MAP.get(raw_ref, raw_ref)

    # Reference size (e.g. 1 m2 / 1 m3 / 1 kg)
    ref_size = pd.to_numeric(epd_row.get(ref_size_col, 1), errors="coerce")
    if pd.isna(ref_size) or ref_size == 0:
        ref_size_eff = 1.0
    else:
        ref_size_eff = float(ref_size)

    # CO2 per reference unit, pick first valid CO2 column
    co2_per_unit = _pick_first_valid_co2(epd_row, co2_cols)

    # Density (for kg-based references)
    density = pd.to_numeric(epd_row.get(density_col), errors="coerce")

    verts = np.array(mesh.verts).reshape(-1, 3)

    if entry["type"] == "layer" and thickness is not None:
        # IFC thickness is assumed to be in metres; convert here if needed
        thickness_m = float(thickness)

        area = area_by_ransac_plane(verts)
        volume = area * thickness_m

        if reference == "m2":
            amount = area
        elif reference == "m3":
            amount = volume
        elif reference == "kg":
            if epd_name_col and epd_name_col in epd_row.index:
                print(epd_row[epd_name_col])
            print("REFSIZE", ref_size_eff)
            print("density", density)
            amount = volume * density if not pd.isna(density) else np.nan
        else:
            print(f"Unknown reference unit for {el.Name}: {reference}")
            amount = volume
    else:
        volume = compute_mesh_volume(mesh)
        area = area_by_ransac_plane(verts)

        if reference == "m3":
            amount = volume
        elif reference == "m2":
            amount = area
        elif reference == "kg":
            amount = volume * density if not pd.isna(density) else np.nan
        else:
            print(f"Unknown reference unit for {el.Name}: {reference}")
            amount = volume

    if any(np.isnan(x) for x in [amount, co2_per_unit]) or ref_size_eff == 0:
        co2_total = np.nan
    else:
        co2_total = (amount * co2_per_unit) / ref_size_eff

    return {
        "co2_total":    co2_total,
        "amount":       amount,
        "co2_per_unit": co2_per_unit,
        "density":      density,
        "reference":    reference,
        "volume":       volume,
        "area":         area,
    }

def get_or_create_pset(ifc_file, element, pset_name):
    """
    Get an existing IfcPropertySet with name pset_name on the element,
    or create a new one and attach it.
    """
    # Try to reuse existing property set
    for rel in getattr(element, "IsDefinedBy", []):
        if rel.is_a("IfcRelDefinesByProperties"):
            pd = rel.RelatingPropertyDefinition
            if pd.is_a("IfcPropertySet") and pd.Name == pset_name:
                return pd
    # Otherwise create a new property set and relation
    prop_set = ifc_file.create_entity("IfcPropertySet", Name=pset_name, HasProperties=[])
    ifc_file.create_entity(
        "IfcRelDefinesByProperties",
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=element.OwnerHistory,
        RelatedObjects=[element],
        RelatingPropertyDefinition=prop_set
    )
    return prop_set

def set_pset_property(ifc_file, pset, name, value, value_type="IfcReal"):
    """
    Set or overwrite a property in an IfcPropertySet using IfcPropertySingleValue.
    """
    # Remove any existing property with the same name
    props = [p for p in getattr(pset, "HasProperties", []) if p.Name != name]

    # Create the correct IfcValue subtype positionally so the wrapper
    # picks up the single argument correctly
    if value_type == "IfcReal":
        val_ent = ifc_file.create_entity("IfcReal", float(value))
    elif value_type == "IfcIdentifier":
        val_ent = ifc_file.create_entity("IfcIdentifier", str(value))
    elif value_type == "IfcLabel":
        val_ent = ifc_file.create_entity("IfcLabel", str(value))
    else:
        # Fallback for any other simple value types
        val_ent = ifc_file.create_entity(value_type, value)

    # Create the single-value property
    prop = ifc_file.create_entity(
        "IfcPropertySingleValue",
        Name=name,
        NominalValue=val_ent,
        Unit=None
    )
    props.append(prop)

    # Re-attach the updated property list to the property set
    pset.HasProperties = props

def add_epd_assignment_pset(ifc_file, element, uuid, results, epd_name=None, confidence=None):
    """
    Attach an 'EPD Assignment' property set to the element with CO2/EPD information.
    """
    pset = get_or_create_pset(ifc_file, element, "EPD Assignment")

    props = {
        "UUID":                 (uuid,                    "IfcIdentifier"),
        "EPD Name":             (epd_name,                "IfcLabel") if epd_name is not None else None,
        "Match Confidence":     (confidence,              "IfcReal")  if confidence is not None else None,
        "CO2 Equivalent Total": (results["co2_total"],    "IfcReal"),
        "Reference Quantity":   (results["amount"],       "IfcReal"),
        "Reference Type":       (results["reference"],    "IfcLabel"),
        "Based Volume":         (results["volume"],       "IfcReal"),
        "Based Area":           (results["area"],         "IfcReal"),
        "Based Co2 Equivalent": (results["co2_per_unit"], "IfcReal"),
        "Density":              (results["density"],      "IfcReal"),
    }

    for name, entry in props.items():
        if entry is None:
            continue
        val, typ = entry
        # Skip NaN/None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        set_pset_property(ifc_file, pset, name, val, typ)

    return element

def main(ifc_file_path, epd_csv_path, matched_epds_path, output_path, epd_config=None):
    """
    Main entry point to compute CO2 from matched EPDs and write them into the IFC.

    Args:
        ifc_file_path:     path to input IFC file
        epd_csv_path:      path to EPD CSV file
        matched_epds_path: path to matched_epds.npy (results from the matching step)
        output_path:       path to write updated IFC with EPD properties
        epd_config:        dict defining column names and mappings for the EPD dataset
                           (if None, DEFAULT_EPD_CONFIG is used)
    """
    if epd_config is None:
        epd_config = DEFAULT_EPD_CONFIG

    skip_entities = ["IfcRailing"]
    layer_results = []

    matched = np.load(matched_epds_path, allow_pickle=True)
    ifc_file = ifcopenshell.open(ifc_file_path)
    elements_by_guid = {el.GlobalId: el for el in ifc_file.by_type("IfcElement")}
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    epd_df = pd.read_csv(
        epd_csv_path,
        sep=";",           # could also be made configurable if needed
        dtype=str,
        decimal=",",
        thousands=".",
        encoding="latin1",
        low_memory=False
    )

    # Numeric columns are inferred from the EPD configuration
    numeric_cols = set(epd_config["co2_cols"] + [epd_config["density_col"]])
    for col in numeric_cols:
        if col not in epd_df.columns:
            continue
        epd_df[col] = (
            epd_df[col]
            .astype(str)
            # 1) remove dots used as thousands separators
            .str.replace(r"\.", "", regex=True)
            # 2) convert decimal comma to dot
            .str.replace(",", ".", regex=False)
            # 3) parse as float, invalid values become NaN
            .pipe(pd.to_numeric, errors="coerce")
        )

    for entry in matched:
        uuid = entry["match"].get("UUID")
        if not uuid:
            print("No UUID for entry:", entry)
            continue

        epd_row = get_epd_row_by_uuid(epd_df, uuid, epd_config)
        if epd_row is None:
            print("UUID not found in EPD CSV:", uuid)
            continue

        for guid in entry["guids"]:
            el = elements_by_guid.get(guid)
            if el is None:
                print("GUID not in IFC:", guid)
                continue
            if any(el.is_a(cls) for cls in skip_entities):
                print(f"Skipping {el.GlobalId} as {el.is_a.__self__}")
                continue

            print(f"Processing {getattr(el, 'Name', '')} ({guid})")
            try:
                shape = ifcopenshell.geom.create_shape(settings, el)
                mesh  = shape.geometry
            except Exception as e:
                print("Geometry failed:", e)
                continue

            if entry["type"] == "layer":
                layers, usage = get_material_layers(el)
                idx = (entry.get("layer_index") or 1) - 1
                if not layers or idx < 0 or idx >= len(layers):
                    print(f"Bad layer index {entry.get('layer_index')} for {guid}")
                    continue
                layer = layers[idx]
                thickness = layer.LayerThickness
                res = calculate_quantity_and_co2(
                    el, entry, mesh, epd_row, epd_config, layer, thickness
                )
                layer_results.append({
                    "GlobalId":    guid,
                    "UUID":        uuid,
                    "layer_index": entry.get("layer_index"),
                    "layer_name":  layer.Name if hasattr(layer, "Name") else f"layer_{idx + 1}",
                    "area_m2":     np.round(res["area"], 3),
                    "volume_m3":   np.round(res["volume"], 3),
                    "co2_total":   np.round(res["co2_total"], 3)
                })
            else:
                res = calculate_quantity_and_co2(el, entry, mesh, epd_row, epd_config)

            epd_name   = entry["match"].get("Name")        # descriptive EPD name from the matching JSON
            confidence = entry["match"].get("confidence")  # matching confidence (if provided)

            add_epd_assignment_pset(ifc_file, el, uuid, res, epd_name, confidence)
            print(f" → CO2eq={res['co2_total']:.2f}, {res['reference']}={res['amount']:.3f}")

    ifc_file.write(output_path)
    print("Wrote updated IFC to", output_path)
    # Write per-layer results to CSV for further analysis

    import csv
    with open(os.path.join(output_path,"computed_layer_co2.csv"), "w", newline="") as f:
        fieldnames = ["GlobalId", "UUID", "layer_index", "layer_name", "area_m2", "volume_m3", "co2_total"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(layer_results)

if __name__ == "__main__":
    base_dir = ""
    ifc_file_path     = os.path.join(base_dir, "Examples/Ifcs/Duplex_A.ifc")
    epd_csv_path      = os.path.join(base_dir, "Examples/Epds/OBD_2024_I_2025-05-20T08_59_27.csv")
    matched_epds_path = os.path.join(base_dir, "outputs/matched_epds.npy")
    output_path       = os.path.join(base_dir, "Duplex_A_with_epd.ifc")

    # Use the default configuration (Ökobau dataset)
    main(ifc_file_path, epd_csv_path, matched_epds_path, output_path)

    # Example for a custom EPD dataset:
    # custom_config = {
    #     "uuid_col":     "epd_uuid",
    #     "ref_unit_col": "unit",
    #     "ref_size_col": "unit_size",
    #     "co2_cols":     ["GWP_total", "GWP_A1_A3"],
    #     "density_col":  "density_kg_m3",
    #     "epd_name_col": "epd_name",
    # }
    # main(ifc_file_path, epd_csv_path, matched_epds_path, output_path, epd_config=custom_config)
