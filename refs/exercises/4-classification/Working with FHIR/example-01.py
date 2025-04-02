from fhir.resources.patient import Patient
import json

# Load a Patient Resource (an animal!)

# Opening JSON file
f = open('./fhir-resources/patient-example-animal.json')

# returns JSON object as
# a dictionary

# Load the FHIR Data
fhir_data = json.load(f)
# Create a Patient object from the JSON data
patient = Patient(**fhir_data)

# Extract specific fields
patient_name = patient.name[0].given[0]
species = patient.extension[0].extension[0].valueCodeableConcept.coding[0].display
breed = patient.extension[0].extension[1].valueCodeableConcept.coding[0].display
gender_status = patient.extension[0].extension[2].valueCodeableConcept.coding[0].code

print(f"Patient Name: {patient_name}")
print(f"Species: {species}")
print(f"Breed: {breed}")
print(f"Gender Status: {gender_status}")


# 2. Update Certain Fields

# Update the patient's contact number
patient.contact[0].telecom[0].value = "(03) 9999 9999"

# Update the managing organization
patient.managingOrganization.display = "New Veterinary Services"

# Convert back to a FHIR model if needed to send or store
updated_fhir_json = patient.json(indent=2)


# 3. Validate the Presence of Required Fields
# The fhir.resources library automatically validates required fields when parsing the JSON data into a FHIR model, throwing an error if a required field is missing. However, if you want to explicitly check for the presence of optional fields, you could do so as follows:

# Example: Check if 'managingOrganization' is present
is_valid = 'managingOrganization' in patient.dict()

print(f"Validation Result: {'Valid' if is_valid else 'Invalid'}")


# 4. Generate a Human-readable Summary
# Finally, you can generate a human-readable summary of the patient information using the FHIR model's attributes:

# Generate a summary of the patient information
summary = f"""
Patient ID: {patient.id}
Name: {patient_name}
Species: {species}
Breed: {breed}
Gender: {patient.gender}
Birth Date: {patient.birthDate}
Owner: {' '.join(patient.contact[0].name.given)} {patient.contact[0].name.family}
Contact: {patient.contact[0].telecom[0].value}
Managing Organization: {patient.managingOrganization.display}
"""

print(summary)




# Closing file
f.close()