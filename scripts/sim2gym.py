import xml.etree.ElementTree as ET


def add_inertial(link, mass=1.0, inertia=None):
    """Add an <inertial> block to a link."""
    inertial = ET.SubElement(link, "inertial")
    
    # Mass
    mass_elem = ET.SubElement(inertial, "mass")
    mass_elem.set("value", str(mass))
    
    # Origin
    origin_elem = ET.SubElement(inertial, "origin")
    origin_elem.set("xyz", "0. 0. 0.")
    origin_elem.set("rpy", "0. 0. 0.")
    
    # Inertia
    inertia_elem = ET.SubElement(inertial, "inertia")
    if inertia is None:
        inertia = {"ixx": "1.0", "ixy": "0.0", "ixz": "0.0", "iyy": "1.0", "iyz": "0.0", "izz": "1.0"}
    for key, value in inertia.items():
        inertia_elem.set(key, value)


def modify_urdf(input_file, output_file):
    """Modify URDF to make it compatible with Isaac Gym."""
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Iterate over all links
    for link in root.findall("link"):
        # Add <inertial> tag if missing
        if link.find("inertial") is None:
            add_inertial(link, mass=5.0)
    
    # Iterate over all joints
    for joint in root.findall("joint"):
        # Remove 'effort' and 'velocity' attributes from <limit> if they exist
        limit = joint.find("limit")
        if limit is not None:
            for attr in ["effort", "velocity"]:
                if attr in limit.attrib:
                    del limit.attrib[attr]

    # Write back the modified URDF
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Modified URDF saved to: {output_file}")


# Input and output files
input_urdf = "car.urdf"  # Replace with the input URDF file path
output_urdf = "car_gym.urdf"  # Replace with the desired output URDF file path

# Modify the URDF
modify_urdf(input_urdf, output_urdf)
