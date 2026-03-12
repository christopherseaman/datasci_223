# Debugging Hints

## General Approach
1. Start by running the script with the sample data
2. Compare the output with the expected output in the documentation
3. Use a debugger to inspect values at key calculation points
4. Pay attention to error messages - they often point to the general area of the bug

## Progressive Hints

### Hint 1: Data Loading
- Check if the data is being loaded correctly
- Are all fields present in the expected format?
- Is the data path correct?

### Hint 2: Medication Names
- How are medication names being handled?
- Check for case sensitivity
- Look for typos in medication names
- Are the medications in LOADING_DOSE_MEDICATIONS matching DOSAGE_FACTORS?

### Hint 3: Calculations
- Review the dosage calculation formula in the documentation
- What mathematical operations are being used?
- Are loading doses being calculated correctly?
- Is the total medication sum working?

### Hint 4: Data Structures
- Are lists and dictionaries properly formatted?
- Check for syntax errors in data structures
- Are we modifying the right dictionaries?

### Hint 5: Output
- Compare each field in the output with the documentation
- Are warnings being generated correctly?
- Is loading dose information accurate?

## Testing Tips
1. Use small, known inputs first
2. Test each medication type separately
3. Test both first doses and regular doses
4. Verify total medication calculations manually
5. Check warning messages for each medication type 