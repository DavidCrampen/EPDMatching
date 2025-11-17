import ifcopenshell
import ifcopenshell.util.element as element_utils

# Paths
def color_on_confidence(ifc_file_path, output_path):
    # Open IFC
    ifcfile = ifcopenshell.open(ifc_file_path)

    # Define color constructors
    def define_color(color_rgb_255, name="Color",ifcfile=ifcfile):
        # Normalize values to [0.0, 1.0]
        color_rgb = [v / 255.0 for v in color_rgb_255]
        col = ifcfile.createIfcColourRgb(Name=name, Red=color_rgb[0], Green=color_rgb[1], Blue=color_rgb[2])
        rendering = ifcfile.createIfcSurfaceStyleRendering(
            SurfaceColour=col,
            ReflectanceMethod="FLAT"
        )
        surface_style = ifcfile.create_entity("IfcSurfaceStyle", Side="BOTH", Styles=[rendering])
        return surface_style

    def set_color(style, element, ifcfile):
        try:
            if not hasattr(element, "Representation") or not element.Representation:
                return

            for rep in element.Representation.Representations:
                if hasattr(rep, "Items") and len(rep.Items) > 0:
                    for item in rep.Items:
                        try:
                            # Attach IfcStyledItem
                            styled_item = ifcfile.create_entity("IfcStyledItem", Item=item, Styles=[style])

                            # Assign it explicitly (important for some viewers)
                            item.StyledByItem = styled_item

                            return
                        except Exception as e:
                            print(f"⚠️ Inner style error: {e}")
                            continue

            print(f"⚠️ No styled item attached for {element.GlobalId}")

        except Exception as e:
            print(f"⚠️ Failed to set color for {element.GlobalId}: {e}")

    # Define RGB colors (normalized 0–1)
    style_cache = {}

    def get_or_create_style(rgb255, name):
        key = tuple(rgb255)
        if key not in style_cache:
            style_cache[key] = define_color(rgb255, name)
        return style_cache[key]

    red    = define_color([220,20,60], "Red",ifcfile)
    yellow = define_color([255,109,0], "Yellow",ifcfile)
    green  = define_color([124.,252.,0.], "Green",ifcfile)
    blue   = define_color([0.,191.,255.], "Blue",ifcfile)

    # Loop over elements
    all_conf= []
    style = None
    colored_count = 0
    for element in ifcfile.by_type("IfcElement"):
        try:
            psets = element_utils.get_psets(element)
            epd_pset = psets.get("EPD Assignment")
            if not epd_pset:
                continue

            confidence_raw = epd_pset.get("Match Confidence")
            if confidence_raw is not None:
                try:
                    confidence = float(str(confidence_raw).strip().replace(",", "."))
                except ValueError:
                    confidence = None
            else:
                confidence = None
            if confidence is not None:
                confidence = float(confidence)
                all_conf.append(confidence)
                if confidence < 65.:
                    style = red
                elif confidence < 85.:
                    style = yellow
                elif confidence >= 85.:
                    style = green
            else:

                style = blue

            print(confidence)
            print(style)
            set_color(style, element,ifcfile)
            colored_count += 1

        except Exception as e:
            print(f"Failed on {element.GlobalId}: {e}")
    import numpy as np
    # Save output
    ifcfile.write(output_path)
    print(f"✅ Colored {colored_count} elements based on confidence.\n💾 Saved to: {output_path}")
    print(np.unique(np.array(all_conf), return_counts = True))


if __name__ == "__main__":
    input_path = "outputs/Duplex_A_with_epd.ifc"
    output_path = "outputs/EPD_Duplex_colored_with_EPD.ifc"