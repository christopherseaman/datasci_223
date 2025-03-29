from fhir.resources.patient import Patient
from jsonschema import validate
import json

# FHIR Schema Validation example


schema_file = open('./fhir.schema.json')
patient_file = open('./fhir-resources/patient-example-animal.json')

fhir_schema = json.load(schema_file)
patient_resource = json.load(patient_file)

schema_file.close()
patient_file.close()

validate(instance=patient_resource, schema=fhir_schema)


invalid_fhir_resource = {
    "resourceType": "Patient2",
}

try:
    validate(instance=invalid_fhir_resource, schema=fhir_schema)
except Exception as e:
    print("Invalid FHIR Resource")
