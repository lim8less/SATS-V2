import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Load predictions from JSON file
with open('predictions.json', 'r') as f:
    data = json.load(f)
    junction_ids = data['junction_ids']
    predictions_dict = data['predictions']

# Function to generate phase states dynamically
def generate_phase_states(num_phases):
    phases = []
    for i in range(num_phases):
        if i % 3 == 0:
            phases.append("GGgrrrGGgrrr")
        elif i % 3 == 1:
            phases.append("yyyrrryyyrrr")
        else:
            phases.append("rrrGGgrrrGGg")
    return phases

# Function to create traffic light logic XML file based on predictions
def create_traffic_light_logic(predictions_dict, output_file='traffic_light_logic.xml'):
    root = ET.Element("additional")

    for idx, (junction_id, predictions) in enumerate(predictions_dict.items()):
        # Ensure unique programID by combining index with junction_id
        program_id = f"{junction_id}_{idx}"
        tl_logic = ET.SubElement(root, "tlLogic", id=junction_id, type="static", programID=program_id, offset="0")

        num_phases = len(predictions)
        phases = generate_phase_states(num_phases)
        valid_phases_added = False

        for i, duration in enumerate(predictions):
            print(f"Junction ID: {junction_id}, Prediction[{i}]: {predictions[i]}")
            try:
                duration = int(float(duration))
                if duration <= 0:
                    print(f"Skipping invalid prediction for junction {junction_id} at index {i}: {predictions[i]} (Non-positive duration)")
                    continue
                
                state = phases[i % len(phases)]
                ET.SubElement(tl_logic, "phase", duration=str(duration), state=state)
                valid_phases_added = True
            except (ValueError, TypeError) as e:
                print(f"Skipping invalid prediction for junction {junction_id} at index {i}: {predictions[i]} ({e})")
                continue

        if not valid_phases_added:
            print(f"Adding default phases for junction {junction_id} due to lack of valid predictions.")
            ET.SubElement(tl_logic, "phase", duration="42", state="GGgrrrGGgrrr")  # Default Green phase for main direction
            ET.SubElement(tl_logic, "phase", duration="3", state="yyyrrryyyrrr")   # Default Yellow phase for main direction
            ET.SubElement(tl_logic, "phase", duration="42", state="rrrGGgrrrGGg")  # Default Green phase for secondary direction
            ET.SubElement(tl_logic, "phase", duration="3", state="rrryyyrrryyy")   # Default Yellow phase for secondary direction

    # Convert to string and pretty print
    xml_str = ET.tostring(root, encoding='utf-8')
    parsed = minidom.parseString(xml_str)
    pretty_xml_str = parsed.toprettyxml(indent="    ")

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml_str)

# Create the traffic light logic file
create_traffic_light_logic(predictions_dict)
