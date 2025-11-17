import ifcopenshell
import numpy as np
import io
import time
import openai
import re
import json
import os
from tqdm import tqdm


import statistics
from collections import Counter

import hashlib

def hash_prompt(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def clean_name(name: str) -> str:
    if not name:
        return None
    # Remove all trailing colon-number sequences like :123, :456, etc.
    return re.sub(r'(:\d+)+$', '', name.strip()).lower()


def get_pset_value(element, pset_name: str, prop_name: str):
    for rel in getattr(element, 'IsDefinedBy', []):
        if rel.is_a('IfcRelDefinesByProperties'):
            pd = rel.RelatingPropertyDefinition
            if pd.is_a('IfcPropertySet') and pd.Name == pset_name:
                for prop in pd.HasProperties:
                    if prop.Name == prop_name and hasattr(prop, 'NominalValue'):
                        val = prop.NominalValue
                        return getattr(val, 'wrappedValue', val)
    return None

def group_elements_and_layers(ifc_file_path, exclude_types=None):
    """
    Returns:
      - element_groups: dict[(clean_name, classification_desc)] -> list[IfcBuildingElement]
      - layer_groups:   dict[(clean_layer_name, material_name, thickness)] -> list[(element, layer_idx, layer_obj)]
    """
    model = ifcopenshell.open(ifc_file_path)
    exclude_types = set(t.lower() for t in (exclude_types or []))
    element_groups: dict = {}
    layer_groups: dict = {}

    for element in model.by_type('IfcBuildingElement'):
        if element.is_a().lower() in exclude_types:
            continue

        # --- MaterialLayers finden (Usage oder Set) ---
        layers = None
        for rel in getattr(element, 'HasAssociations', []) or []:
            if not rel.is_a('IfcRelAssociatesMaterial'):
                continue
            mat = rel.RelatingMaterial
            if mat.is_a('IfcMaterialLayerSetUsage'):
                layers = list(mat.ForLayerSet.MaterialLayers)
                break
            elif mat.is_a('IfcMaterialLayerSet'):
                layers = list(mat.MaterialLayers)
                break

        if layers:
            # --- Element mit Layern → in layer_groups packen ---
            for idx, layer in enumerate(layers):
                # Robuster Layer-Name
                raw_layer_name = None
                if getattr(layer, "Name", None):
                    raw_layer_name = layer.Name
                elif getattr(layer, "Category", None):
                    raw_layer_name = layer.Category
                elif getattr(layer, "Material", None) and getattr(layer.Material, "Name", None):
                    raw_layer_name = layer.Material.Name

                layer_clean = clean_name(raw_layer_name) if raw_layer_name else None

                material_name = None
                if getattr(layer, "Material", None) and getattr(layer.Material, "Name", None):
                    material_name = layer.Material.Name.strip().lower()

                # LayerThickness kann auch None oder 0 sein → float + leicht runden
                thickness_val = getattr(layer, "LayerThickness", 0) or 0.0
                thickness = round(float(thickness_val), 4)

                key = (layer_clean, material_name, thickness)
                layer_groups.setdefault(key, []).append((element, idx, layer))
        else:
            # --- Element ohne Layer → in element_groups packen ---
            name_clean = clean_name(getattr(element, "Name", None))
            classification_desc = get_pset_value(element, "PSet_Revit_Type_Other", "Classification Description")
            key = (name_clean, classification_desc)
            element_groups.setdefault(key, []).append(element)

    return element_groups, layer_groups



def create_element_prompts_grouped(element_groups):
    prompts = []

    for (name_clean, classification_desc), elements in element_groups.items():
        elem = elements[0]
        guids = [e.GlobalId for e in elements]
        buffer = io.StringIO()

        # Collect statistics
        levels = [getattr(e.Storey, 'Name', None) for e in elements if getattr(e, 'Storey', None)]
        external_flags = [get_pset_value(e, 'Pset_BuildingElementCommon', 'IsExternal') for e in elements]
        heights = [getattr(e, 'OverallHeight', None) for e in elements if getattr(e, 'OverallHeight', None)]
        widths = [getattr(e, 'OverallWidth', None) for e in elements if getattr(e, 'OverallWidth', None)]
        depths = [getattr(e, 'OverallDepth', None) for e in elements if getattr(e, 'OverallDepth', None)]

        print("Please match an EPD to the following IFC building element group with these properties:", file=buffer)
        print(f"- Group size: {len(elements)} elements", file=buffer)
        print(f"- IfcType: {elem.is_a()}", file=buffer)
        print(f"  - Representative GlobalId: {elem.GlobalId}", file=buffer)
        if elem.Name:
            print(f"  - Name: {elem.Name}", file=buffer)
        if elem.ObjectType:
            print(f"  - ObjectType: {elem.ObjectType}", file=buffer)
        if elem.Tag:
            print(f"  - Tag: {elem.Tag}", file=buffer)

        # Classification
        if classification_desc:
            print(f"  - ClassificationDescription: {classification_desc}", file=buffer)
        print(f"  - CleanName: {name_clean}", file=buffer)

        # Aggregated levels
        level_counts = Counter(levels)
        if level_counts:
            print(f"  - StoreyLevels: {dict(level_counts)}", file=buffer)

        # External flag stats
        ext_counts = Counter(str(flag) for flag in external_flags if flag is not None)
        if ext_counts:
            print(f"  - IsExternal distribution: {dict(ext_counts)}", file=buffer)

        # Geometry stats
        if heights:
            print(f"  - Height range: {min(heights):.3f}–{max(heights):.3f} m", file=buffer)
        if widths:
            print(f"  - Width range: {min(widths):.3f}–{max(widths):.3f} m", file=buffer)
        if depths:
            print(f"  - Depth range: {min(depths):.3f}–{max(depths):.3f} m", file=buffer)



        prompts.append((buffer.getvalue(), guids, {'type': 'element_group', 'classification': classification_desc}))

    return prompts

def create_layer_prompts_grouped(layer_groups):
    prompts = []

    for (layer_clean, material_name, thickness), items in layer_groups.items():
        buffer = io.StringIO()

        # Repräsentativer Eintrag
        elem, idx, layer = items[0]
        parent_guids  = [e.GlobalId for e, _, _ in items]
        parent_types  = [e.is_a() for e, _, _ in items]

        # Storeys über alle Eltern
        storeys = []
        external_flags = []
        for e, _, _ in items:
            if getattr(e, "Storey", None):
                storeys.append(getattr(e.Storey, "Name", None))
            ext_val = get_pset_value(e, "Pset_BuildingElementCommon", "IsExternal")
            if ext_val is not None:
                external_flags.append(ext_val)

        thicknesses = [float(getattr(l, 'LayerThickness', 0) or 0.0) for _, _, l in items]
        categories = [getattr(l, 'Category', None) for _, _, l in items if getattr(l, 'Category', None)]

        material_names = []
        descriptions = []
        for _, _, l in items:
            mat = getattr(l, "Material", None)
            if mat:
                if getattr(mat, "Name", None):
                    material_names.append(mat.Name)
                if getattr(mat, "Description", None):
                    descriptions.append(mat.Description)

        # --- Prompt bauen ---
        print("Please match an EPD to the following material layer, based on its description and usage context.", file=buffer)
        print("This layer appears in multiple IFC building elements. The following information summarizes its use:", file=buffer)
        print(f"- Group size: {len(items)} layer instances", file=buffer)

        # Parent type summary
        type_counts = Counter(parent_types)
        print(f"- ParentElementTypes: {dict(type_counts)}", file=buffer)

        # Storey levels
        if storeys:
            print(f"- Storey distribution: {dict(Counter(storeys))}", file=buffer)

        # IsExternal summary
        if external_flags:
            print(f"- IsExternal distribution: {dict(Counter(str(f) for f in external_flags))}", file=buffer)

        # Layer properties (das eigentliche Match-Objekt)
        print("Layer-specific information (this is the object to be matched):", file=buffer)
        print(f"  - LayerIndex (in parent stacks): {idx + 1}", file=buffer)
        if layer_clean:
            print(f"  - CleanLayerName: {layer_clean}", file=buffer)
        if material_name:
            print(f"  - MaterialName (canonical): {material_name}", file=buffer)
        if material_names:
            print(f"  - MaterialName variants: {dict(Counter(material_names))}", file=buffer)
        # Dicke
        mean_th = statistics.mean(thicknesses) if thicknesses else 0.0
        print(f"  - LayerThickness (mean): {mean_th:.4f} m", file=buffer)
        if len(set(thicknesses)) > 1:
            print(f"  - Thickness range: {min(thicknesses):.4f}–{max(thicknesses):.4f} m", file=buffer)

        if descriptions:
            desc_counter = Counter(descriptions)
            print(f"  - Material Descriptions: {dict(desc_counter)}", file=buffer)
        if categories:
            print(f"  - Layer Categories: {dict(Counter(categories))}", file=buffer)

        prompts.append((buffer.getvalue(), parent_guids, {
            'type': 'layer',
            'layer_index': idx + 1,      # repräsentativer Index, dein CO2-Skript nutzt das
            'layer_name': layer_clean,
            'material': material_name
        }))

    return prompts

def match_epd(message, assistant_id):
    thread = openai.beta.threads.create()
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message
    )
    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )
    max_wait = 120
    start_time = time.time()
    while True:
        status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id).status
        if status == "completed": break
        if status == "failed": raise Exception("Run failed!")
        if time.time() - start_time > max_wait: raise TimeoutError("Assistant run timed out")
        time.sleep(1)
    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    for msg in reversed(messages.data):
        if msg.role == "assistant": return msg.content[0].text.value
    return None

def parse_json_string(s):
    cleaned = re.sub(r"```json|```", "", s).strip()
    return json.loads(cleaned)

def save_matches(results, filename='matched_epds.npy'):
    np.save(filename, results)

def summarize_element_groups(element_groups):
    summary = {}
    for key, elements in element_groups.items():
        summary[key] = [el.GlobalId for el in elements]
    return summary

def summarize_layer_groups(layer_groups):
    summary = {}
    for key, layer_entries in layer_groups.items():
        # Each entry is (parent_element, layer_idx, layer_obj)
        summary[key] = [
            {
                "element_id": el.GlobalId,
                "layer_index": idx,
                "layer_name": layer.Material.Name if layer.Material and hasattr(layer.Material, "Name") else None
            }
            for el, idx, layer in layer_entries
        ]

    return summary

def run_grouped_matching(ifc_file_path, assistant_id, exclude_types=None, save_path='matched_epds.npy'):
    element_groups, layer_groups = group_elements_and_layers(ifc_file_path, exclude_types)
    element_prompts = create_element_prompts_grouped(element_groups)
    layer_prompts   = create_layer_prompts_grouped(layer_groups)

    # Step 1: collect all GUIDs from layer prompts (Eltern von Layer-Gruppen)
    layer_guid_set = set()
    for _, guids, _ in layer_prompts:
        layer_guid_set.update(guids)

    # Step 2: element_prompts filtern, wenn sie bereits durch Layer-Matches abgedeckt sind
    filtered_element_prompts = [
        (prompt, guids, meta)
        for prompt, guids, meta in element_prompts
        if not any(g in layer_guid_set for g in guids)
    ]

    prompt_cache = {}
    if os.path.exists(save_path):
        existing = np.load(save_path, allow_pickle=True)
        for item in existing:
            h = hash_prompt(item['prompt'])
            prompt_cache[h] = item

    results = []

    for prompt, guids, meta in tqdm(filtered_element_prompts + layer_prompts, desc="Matching EPDs"):
        h = hash_prompt(prompt)

        if h in prompt_cache:
            print("✅ Using cached result")
            results.append(prompt_cache[h])
            continue

        print(prompt)
        response = match_epd(prompt, assistant_id)
        try:
            match = parse_json_string(response)
        except Exception as e:
            print(f"⚠️ Failed to parse response for GUIDs {guids}: {e}")
            match = {'error': 'parse failure', 'raw': response}

        print(match)
        entry = {
            'guids': guids,
            'match': match,
            'type': meta['type'],
            'prompt': prompt
        }
        if meta['type'] == 'layer':
            entry['layer_index'] = meta['layer_index']
        results.append(entry)
        prompt_cache[h] = entry

    save_matches(results, save_path)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    assistant_id = "assistants/your_assistant_id_here"  # <-- replace with your assistant ID

    # Example usage:
    run_grouped_matching(
        ifc_file_path="Duplex_A.ifc",  # <-- replace with your IFC file path
        assistant_id=assistant_id,
        exclude_types=["IfcFurnishingElement", "IfcSpace", "IfcOpeningElement"],
        save_path='matched_epds.npy'
    )
