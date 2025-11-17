import openai
import time
import ifcopenshell
import ifcopenshell.geom
import os
import io
import numpy as np
import re
import tiktoken
from ifcopenshell.util.element import get_material


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

    max_wait = 120  # seconds
    start_time = time.time()

    while True:
        run_status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == "completed":
            break
        elif run_status.status == "failed":
            raise Exception("❌ Run failed!")
        elif time.time() - start_time > max_wait:
            raise TimeoutError("⏳ Assistant run timed out after 120 seconds.")
        time.sleep(1)

    messages = openai.beta.threads.messages.list(thread_id=thread.id)

    for msg in reversed(messages.data):
        if msg.role == "assistant":
            response = msg.content[0].text.value
            print("\n✅ Assistant Response:")
            print(response)
            return response
    return None


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def describe_ifc_elements_unique(ifc_path: str, exclude_types: list[str] = None):
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    request_count = 0
    ifc_file = ifcopenshell.open(ifc_path)
    exclude_types = set(exclude_types or [])
    messages = []
    guids_all = []
    matches = []
    seen_fingerprints = {}
    layer_match_cache = {}  # key = (name, material name lower), value = (message, match)

    IGNORED_PROPSETS = {"PSet_Revit_Type_Other", "PSet_Revit_Structural Analysis", "PSet_Revit_Analytical Model",
                        "PSet_Revit_Constraints", "PSet_Revit_Type_Structural", "PSet_Revit_Type_Dimensions",
                        "PSet_Revit_Structural", "PSet_Revit_Other"}

    def get_bounding_box_dims(product):
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            bbox = shape.geometry.bbox()
            width = round(bbox.max.x - bbox.min.x, 3)
            height = round(bbox.max.z - bbox.min.z, 3)
            depth = round(bbox.max.y - bbox.min.y, 3)
            return width, height, depth
        except Exception:
            return None, None, None

    def get_property_sets(element):
        prop_sets = {}
        for rel in element.IsDefinedBy:
            if rel.is_a("IfcRelDefinesByProperties"):
                prop_def = rel.RelatingPropertyDefinition
                if prop_def.is_a("IfcPropertySet") and prop_def.Name not in IGNORED_PROPSETS:
                    props = {}
                    for prop in prop_def.HasProperties:
                        val = getattr(prop, "NominalValue", None)
                        if val:
                            val = val.wrappedValue if hasattr(val, "wrappedValue") else val
                        if val != prop.Name:
                            props[prop.Name] = val
                    if props:
                        prop_sets[prop_def.Name] = props
        return prop_sets

    def get_level_name(element):
        for rel in element.ContainedInStructure or []:
            if rel.is_a("IfcRelContainedInSpatialStructure"):
                return getattr(rel.RelatingStructure, "Name", "Unknown")
        return "Unknown"

    for element in ifc_file.by_type("IfcElement"):
        element_type = element.is_a()
        if element_type in exclude_types:
            continue

        name = getattr(element, "Name", None)
        obj_type = getattr(element, "ObjectType", None)
        tag = getattr(element, "Tag", None)
        guid = element.GlobalId
        width, height, depth = get_bounding_box_dims(element)

        fingerprint = (element_type, name, obj_type)
        if fingerprint in seen_fingerprints:
            seen_fingerprints[fingerprint]["guids"].append(guid)
            continue

        level = get_level_name(element)
        material = get_material(element)

        if material and material.is_a("IfcMaterialLayerSetUsage"):
            layer_set = material.ForLayerSet
            for idx, layer in enumerate(layer_set.MaterialLayers):
                mat_name = layer.Material.Name.lower().strip() if layer.Material and layer.Material.Name else f"Unnamed_{idx+1}"
                layer_thickness = layer.LayerThickness
                layer_key = (name, mat_name)

                if layer_key in layer_match_cache:
                    print(f"🔁 Reusing cached match for {layer_key}")
                    messages.append(layer_match_cache[layer_key][0])
                    matches.append(layer_match_cache[layer_key][1])
                    guids_all.append([guid, f"layer_{idx+1}"])
                    continue

                layer_buffer = io.StringIO()
                print(
                    "Please match an EPD to the following material layer, which is part of an IFC building element described below:",
                    file=layer_buffer)
                print(f"- ParentElementType: {element_type}", file=layer_buffer)
                print(f"  - GlobalId: {guid}", file=layer_buffer)
                if name: print(f"  - Name: {name}", file=layer_buffer)
                if level: print(f"  - Level: {level}", file=layer_buffer)
                print(f"  - LayerIndex: {idx+1}", file=layer_buffer)
                print(f"  - LayerName: {mat_name}", file=layer_buffer)
                print(f"  - LayerThickness: {layer_thickness} m", file=layer_buffer)
                if height: print(f"  - EstimatedHeight: {height}", file=layer_buffer)
                if width: print(f"  - EstimatedWidth: {width}", file=layer_buffer)
                if depth: print(f"  - EstimatedDepth: {depth}", file=layer_buffer)

                layer_msg = layer_buffer.getvalue()
                print(layer_msg)
                msg_tokens = estimate_tokens(layer_msg)
                print(f"📏 Token count: {msg_tokens}")

                try:
                    match = match_epd(layer_msg, assistant_id=assistant_id)
                    time.sleep(1)
                    request_count+=1
                    print(f"Request Number: {request_count} done!")

                    matches.append(match)
                    messages.append(layer_msg)
                    guids_all.append([guid, f"layer_{idx+1}"])
                    layer_match_cache[layer_key] = (layer_msg, match)
                except Exception:
                    print(f"❌ Matching failed for {guid} layer {idx+1}")
        else:
            base_buffer = io.StringIO()
            print("Please match an EPD to the following ifc entity with its properties:", file=base_buffer)
            print(f"- IfcType: {element_type}", file=base_buffer)
            print(f"  - GlobalId: {guid}", file=base_buffer)
            for attr_name, attr_value in [("Name", name), ("ObjectType", obj_type), ("Tag", tag)]:
                if attr_value:
                    print(f"  - {attr_name}: {attr_value}", file=base_buffer)
            if height: print(f"  - EstimatedHeight: {height}", file=base_buffer)
            if width: print(f"  - EstimatedWidth: {width}", file=base_buffer)
            if depth: print(f"  - EstimatedDepth: {depth}", file=base_buffer)

            prop_sets = get_property_sets(element)
            for pset_name, props in prop_sets.items():
                print(f"  - PropertySet: {pset_name}", file=base_buffer)
                for k, v in props.items():
                    print(f"    - {k}: {v}", file=base_buffer)
            if level:
                print(f"  - Level: {level}", file=base_buffer)

            base_msg = base_buffer.getvalue()
            print(base_msg)
            msg_tokens = estimate_tokens(base_msg)
            print(f"📏 Token count: {msg_tokens}")
            try:
                match = match_epd(base_msg, assistant_id=assistant_id)
                time.sleep(1)
                request_count += 1
                print(f"Request Number: {request_count} done!")
                matches.append(match)
                messages.append(base_msg)
                guids_all.append([guid])
            except Exception:
                print(f"❌ Matching failed for {guid}")

    return messages, guids_all, matches


if __name__ == "__main__":
    openai.api_key = "-insert your openai api key here-"
    assistant_id = "- insert your assistant id here-"

    messages, guids, matches = describe_ifc_elements_unique(
        ifc_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "Duplex_A.ifc"),
        exclude_types=["IfcFurnishingElement", "IfcSpace", "IfcOpeningElement"]
    )

    np.save("Examples/outputs/messages.npy", messages)
    np.save("Examples/outputs/guids_dict.npy", {i: g for i, g in enumerate(guids)})
    np.save("Examples/outputs/matches.npy", matches)
