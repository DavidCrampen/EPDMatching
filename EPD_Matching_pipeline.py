import EPD_utils.EPD_utils as match
import EPD_utils.CO2_from_matches_and_volumes as co2_calc
import openai
import os
import ifc_utils.color_by_confidence as col_conf


openai.api_key = "-insert your API key here-"
assistant_id = "-insert your assistant id here-"

base_dir = "Examples"

# Exchange this with your IFC file path
ifc_file_path = os.path.join(base_dir, "Ifcs/Duplex_A.ifc")

# Exchange this with your EPD_datasheet path
ökobau_csv_path = os.path.join(base_dir, "Epds/OBD_2024_I_2025-05-20T08_59_27.csv")

# Paths to save intermediate and final outputs useful if you don't want to re-run matching (€)
matched_epds_path = os.path.join(base_dir, "outputs/matched_epds.npy")

# Standard path for outputs of IFC with EPD annotations
output_path = os.path.join(base_dir, "outputs/Duplex_A_with_epd.ifc")

# Run matching and save matched EPDs:
match.run_grouped_matching(
    ifc_file_path=ifc_file_path,  # <-- replace with your IFC file path
    assistant_id=assistant_id,
    exclude_types=["IfcFurnishingElement", "IfcSpace", "IfcOpeningElement"],
    save_path=matched_epds_path
)

# To calculate the estimated CO2 emissions and annotate the IFC:
co2_calc.main(ifc_file_path, ökobau_csv_path, matched_epds_path, output_path)

# If you want to color the IFC by confidence levels of the EPD matching:
colored_ifc_path = os.path.join(base_dir,"outputs/EPD_Duplex_colored_with_EPD.ifc")
col_conf.color_on_confidence(output_path,colored_ifc_path )



