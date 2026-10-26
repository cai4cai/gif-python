import argparse
import csv
import xml.etree.ElementTree as ET

import pandas as pd


def parse_xml_file(file_path):
    def parse_element(element):
        # Create an empty dictionary to store the information
        data_dict = {}

        # Iterate over the elements within the current element
        for child in element:
            # Store the tag name as the key in the dictionary
            key = child.tag

            # If the child has nested elements, recursively parse them
            if len(child) > 0:
                value = parse_element(child)
            else:
                # Store the element's text content as the value in the dictionary
                value = child.text.strip() if child.text else ""

            # Update the dictionary
            if key in data_dict:
                # If the key already exists, convert the value to a list
                if not isinstance(data_dict[key], list):
                    data_dict[key] = [data_dict[key]]
                data_dict[key].append(value)
            else:
                data_dict[key] = value

        return data_dict

    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Recursively parse the XML tree
    data_dict = parse_element(root)

    # Return the populated dictionary
    return data_dict


def extract_tissue_info_and_structures_info(xml_dict, orig_GIF_to_consec_labels):
    tissue_dict = {}
    for d in xml_dict["tissues"]["item"][:]:
        tissue_dict[int(d["number"])] = {"name": d["name"]}

    # extract structures information
    structure_dict = {}
    for d in xml_dict["labels"]["item"][:]:
        orig_label = int(d["number"])
        if orig_label not in orig_GIF_to_consec_labels:
            print(f"Original GIF label {orig_label} with name {d['name']} not found in the csv file")
            continue
        consec_label = orig_GIF_to_consec_labels[orig_label]
        structure_dict[consec_label] = {"name": d["name"], "tissues": [int(t) for t in d["tissues"].split(",")]}

    return tissue_dict, structure_dict


parser = argparse.ArgumentParser(description="Convert a GIF labels.xml into tissues_info.csv and structures_info.csv.")
parser.add_argument("--labels-xml", required=True, help="labels.xml of the C++ GIF database")
parser.add_argument(
    "--label-map-csv",
    required=True,
    help="csv mapping the original GIF labels to consecutive labels "
    "(see map_originalGIF_to_consecutive_labels.csv in this folder)",
)
parser.add_argument("--tissues-info-out", required=True, help="output tissues_info.csv")
parser.add_argument("--structures-info-out", required=True, help="output structures_info.csv")
args = parser.parse_args()

xml_path = args.labels_xml
csv_path = args.label_map_csv

xml_dict = parse_xml_file(xml_path)
print(xml_dict)

orig_GIF_to_consec_labels = pd.read_csv(csv_path, index_col="original GIF labels").to_dict()["consecutive labels"]
print(orig_GIF_to_consec_labels)

tissue_dict, structure_dict = extract_tissue_info_and_structures_info(xml_dict, orig_GIF_to_consec_labels)
print(tissue_dict)
print(structure_dict)

print(len(tissue_dict))
print(len(structure_dict))

tissues_info_path = args.tissues_info_out
structures_info_path = args.structures_info_out


tissue_df = pd.DataFrame(tissue_dict).T
# name the index as "label"
tissue_df.index.name = "label"

tissue_df.to_csv(tissues_info_path, quoting=csv.QUOTE_NONE)
print(tissue_df)

struc_df = pd.DataFrame(structure_dict).T
# name the index as "label"
struc_df.index.name = "label"

struc_df.to_csv(structures_info_path)
print(struc_df[20:])
