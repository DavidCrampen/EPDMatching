import os
import re
import json
import numpy as np
import ifcopenshell
import ifcopenshell.api

# --- EPD Assignment into IFC ---
def add_epd_assignments(ifc_path: str,
                        output_path: str,
                        combined: dict,
                        colorize: bool = False):
    """
    Writes EPD matches into the IFC as PSet properties.
    Supports both element-level and layer-level assignments.

    combined: {
       idx: {
         'guids': [...],         # element GUIDs
         'match': {'UUID', 'Name', 'confidence', ...},
         'type': 'element' | 'layer',
         'layer_index': int   # only for 'layer'
       },
       ...
    }
    """
    # Load IFC model
    ifc = ifcopenshell.open(ifc_path)

    # Prepare color styles if requested
    styles = {}
    if colorize:
        colors_rgb = {
            'green': (0.0, 1.0, 0.0),
            'yellow': (1.0, 1.0, 0.0),
            'red': (1.0, 0.0, 0.0)
        }
        def make_style(name, rgb):
            col = ifc.create_entity('IfcColourRgb', Name=name, Red=rgb[0], Green=rgb[1], Blue=rgb[2])
            ssr = ifc.create_entity('IfcSurfaceStyleRendering', SurfaceColour=col)
            return ifc.create_entity('IfcSurfaceStyle', Name=name, Side='BOTH', Styles=[ssr])
        styles = {c: make_style(f"EPD_{c}", rgb) for c, rgb in colors_rgb.items()}

    def get_psets(elem):
        """Return existing property sets on an element."""
        psets = {}
        for rel in getattr(elem, 'IsDefinedBy', []):
            if rel.is_a('IfcRelDefinesByProperties'):
                pd = rel.RelatingPropertyDefinition
                if pd.is_a('IfcPropertySet'):
                    psets[pd.Name] = pd
        return psets

    # Iterate through combined matches
    for entry in combined.values():
        epd = entry.get('match') or {}
        uuid = epd.get('UUID')
        name = epd.get('Name')
        confidence = epd.get('confidence')
        if not uuid:
            continue

        is_layer = (entry.get('type') == 'layer')
        layer_idx = entry.get('layer_index') if is_layer else None

        for guid in entry.get('guids', []):
            elem = ifc.by_guid(guid)
            if not elem:
                print(f"⚠️ Missing element for GUID {guid}")
                continue

            # Determine PSet name per type
            if is_layer:
                pset_name = f"EPD Assignment Layer {layer_idx}"
            else:
                pset_name = "EPD Assignment"

            # Ensure PSet exists
            existing = get_psets(elem)
            if pset_name in existing:
                pset = existing[pset_name]
            else:
                pset = ifcopenshell.api.run(
                    'pset.add_pset', ifc,
                    product=elem, name=pset_name
                )

            # Prepare properties
            props = { 'UUID': uuid }
            if name:
                props['EPD Name'] = name
            if confidence is not None:
                props['Confidence'] = float(confidence)
            if is_layer:
                props['LayerIndex'] = int(layer_idx)

            # Apply properties
            ifcopenshell.api.run(
                'pset.edit_pset', ifc,
                pset=pset, properties=props
            )

            # Optional: colorize
            if colorize and confidence is not None:
                if confidence > 85:
                    col = 'green'
                elif confidence >= 65:
                    col = 'yellow'
                else:
                    col = 'red'
                style = styles[col]

                rep = getattr(elem, 'Representation', None)
                if rep and getattr(rep, 'Representations', None):
                    for r in rep.Representations:
                        for item in r.Items:
                            if 'StyledByItem' in item.get_info():
                                styled = ifc.create_entity(
                                    'IfcStyledItem', Item=item, Styles=[style]
                                )
                                item.StyledByItem = [styled]

    # Write back
    ifc.write(output_path)
    print(f"✅ Saved with EPD assignments: {output_path}")


# --- Example invocation ---
if __name__ == "__main__":
    base = r"Examples"


    raw = np.load(os.path.join(base, "outputs/matched_epds.npy"), allow_pickle=True)
    results_list = raw.tolist()  # now a Python list

    # 2) Turn it into a dict keyed by index, so add_epd_assignments() can do combined.values()
    combined = { i: entry for i, entry in enumerate(results_list) }

    src = os.path.join(base, 'Ifcs/Duplex_A.ifc')
    dst = os.path.join(base, 'outputs/EPD_Duplex_Layers.ifc')

    add_epd_assignments(src, dst, combined, colorize=True)