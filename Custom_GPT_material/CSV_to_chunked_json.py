import csv
import json

def csv_to_json_filtered(csv_file_path, json_file_path, columns_to_include):
    data = []

    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                filtered_row = {key: row[key] for key in columns_to_include if key in row}
                data.append(filtered_row)
    except UnicodeDecodeError:
        with open(csv_file_path, mode='r', encoding='ISO-8859-1') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                filtered_row = {key: row[key] for key in columns_to_include if key in row}
                data.append(filtered_row)

    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f"✅ Converted {csv_file_path} to {json_file_path} with selected columns.")

def get_csv_column_names(csv_file_path):
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            columns = reader.fieldnames
    except UnicodeDecodeError:
        with open(csv_file_path, mode='r', encoding='ISO-8859-1') as csv_file:
            reader = csv.DictReader(csv_file)
            columns = reader.fieldnames

    print("🧾 Columns in CSV:")
    for col in columns:
        print(f"{col}")

    return columns
# Example usage



def split_json_by_records(input_path, output_prefix, records_per_file=200):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i in range(0, len(data), records_per_file):
        chunk = data[i:i + records_per_file]
        out_path = f"{output_prefix}_part_{i // records_per_file + 1}.json"
        with open(out_path, 'w', encoding='utf-8') as out_file:
            json.dump(chunk, out_file, indent=4, ensure_ascii=False)
        print(f"✅ Created {out_path} with {len(chunk)} records")

if __name__ == "__main__":
    # Example usage

    # get_csv_column_names("OBD_2024_I_2025-03-27T13_32_17.csv")
    columns = ["UUID",
               "Name (de)",
               "Name (en)",
               "Kategorie (original)",
               "Referenzfluss-Name",
               "Laenderkennung",
               "Typ",
               "Bezugsgroesse"
               "Bezugseinheit",
               "Schuettdichte (kg/m3)",
               "Flaechengewicht (kg/m2)",
               "Rohdichte (kg/m3)",
               "Schichtdicke (m)",
               "Laengengewicht (kg/m)",
               "Stueckgewicht (kg)",
               "Modul",
               "Szenariobeschreibung"]  # replace with your actual column names


    csv_to_json_filtered("OBD_2024_I_2025-03-27T13_32_17.csv", "EPD_Dataset.json", columns)
    split_json_by_records("EPD_Dataset.json", "filtered_chunk")


