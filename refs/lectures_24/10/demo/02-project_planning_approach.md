# Demo 2: Project Planning Approach - Gene Expression in Transplant Patients

**Scenario:** Your PI just handed you a gene expression dataset from transplant patients with no clear direction, just a deadline. You need to figure out good project options through systematic exploration.

## The Dataset Context

You've been given:
- Gene expression data from 200 transplant patients
- Patient metadata: age, time post-transplant, rejection status, hospital site
- 20,000+ genes measured via RNA-seq
- A vague directive: "Find something interesting for the grant renewal"

## Step 1: PI Interview - Understanding Research Context (5 minutes)

### Initial Questions to Ask Your PI

**Research Background:**
- "What has your lab's previous research focused on in transplant medicine?"
- "What were you hoping to see from this dataset when it was collected?"
- "What would be the best possible outcome from this data for the grant?"

### Hypothetical PI Responses:

**PI Response 1:** "We've been studying rejection mechanisms for 10 years. I want to find biomarkers that can predict rejection before it becomes clinically apparent."

**PI Response 2:** "This is part of a multi-site study. We're interested in understanding why some patients do better at certain hospitals."

**PI Response 3:** "We collected this as part of a larger biorepository. I'm open to any direction that could lead to a high-impact publication."

**Choose Response 1 for this demo**

### Follow-up Questions Based on Response 1:

- "What biomarkers have you looked at before?"
- "How early can we predict rejection with current methods?"
- "Are there specific gene pathways you suspect are involved?"
- "What would make a biomarker clinically useful?"

**PI's Elaboration:** "Current biopsies can only detect rejection after tissue damage starts. If we could predict it 2-3 months earlier using blood tests, that would be revolutionary. I suspect immune activation pathways are key."

## Step 2: Data Exploration - Understanding What You Have (8 minutes)

### Hypothetical Data Schema

```
Patient Metadata (200 patients):
- patient_id: P001, P002, ...
- age: 25-75 years
- time_post_transplant: 0.5-5 years
- rejection_status: No_Rejection (140), Rejection (60)
- hospital_site: Site_A (80), Site_B (70), Site_C (50)
- sample_type: blood, biopsy

Gene Expression (20,000 genes):
- Genes: ENSG00000000001, ENSG00000000002, ...
- Expression levels: Log2(TPM + 1)
- Quality metrics: % mitochondrial genes, total reads
```

### Initial Exploration Questions

1. **Data Quality Assessment:**
   - Are rejection rates similar across sites? (Site_A: 25%, Site_B: 35%, Site_C: 30%)
   - Do we have batch effects? (Samples processed in 3 batches over 2 years)
   - Missing data patterns? (5% missing expression values, random distribution)

2. **Clinical Patterns:**
   - Rejection timing: Early (0-1 year): 40%, Late (1+ years): 60%
   - Age correlation: Older patients slightly higher rejection risk
   - Site differences: Site_B has more complex cases

3. **Expression Data Quality:**
   - Distribution looks normal after log transformation
   - Some genes have very low expression across all samples
   - Potential outlier samples: 3 samples with unusual expression patterns

### Preliminary Findings to Discuss with PI

"I found some interesting patterns. Site_B has higher rejection rates - they might be taking sicker patients. Also, we have good temporal coverage for early prediction. Should we focus on early rejection specifically?"

## Step 3: Literature Search - Informing Your Understanding (5 minutes)

### Key Search Terms Based on Initial Findings

1. **"transplant rejection biomarkers blood RNA-seq"**
   - Recent reviews show most studies use biopsy tissue
   - Blood-based RNA signatures are emerging field
   - 2-3 commercial tests exist but limited validation

2. **"early rejection prediction gene expression"**
   - Most studies focus on detecting ongoing rejection
   - Few studies predict rejection weeks/months ahead
   - Gap in literature for early warning systems

3. **"immune activation pathways transplant"**
   - JAK-STAT, interferon response pathways well established
   - T-cell activation markers consistently upregulated
   - Novel: metabolic reprogramming during rejection

### Literature Summary for PI

"Good news - there's a clear gap in early prediction. Most studies detect ongoing rejection, not future risk. The interferon pathway is consistently involved, but metabolic changes are understudied. This could be novel."

## Step 4: Hypothesis Refinement with PI (7 minutes)

### Present Options to PI

**Option 1: Early Rejection Prediction Model**
- Focus: Predict rejection 2-3 months before clinical detection
- Methods: Machine learning with temporal analysis
- Timeline: 6 months for initial results
- Clinical impact: Could enable preventive interventions

**Option 2: Site-Specific Rejection Mechanisms**
- Focus: Understand why Site_B has higher rejection rates
- Methods: Differential expression + pathway analysis
- Timeline: 3-4 months
- Clinical impact: Improve patient selection/treatment protocols

**Option 3: Novel Metabolic Pathway Discovery**
- Focus: Identify new biological mechanisms
- Methods: Network analysis + functional validation
- Timeline: 12-18 months
- Clinical impact: New therapeutic targets

### PI Feedback Session

**PI:** "Option 1 sounds most fundable, but is 200 patients enough for machine learning?"

**Your response:** "We could start with a simpler approach - identify the strongest individual biomarkers first, then build complexity. Also check if we can access the larger multi-site dataset you mentioned."

**PI:** "Good point. And for the grant, we need preliminary data in 3 months. What's the minimum viable analysis?"

**Refined Plan:**
1. **Month 1:** Identify top 20 genes associated with future rejection
2. **Month 2:** Validate in cross-validation, test simple prediction model
3. **Month 3:** Generate preliminary figures for grant, plan larger validation study

## Step 5: Next Investigative Cycle (3 minutes)

### Specific Next Steps

1. **Technical Questions:**
   - Should we include patients with borderline rejection?
   - Do we need to account for immunosuppressive medications?
   - How do we handle time-varying gene expression?

2. **Resource Planning:**
   - Need bioinformatics support for pathway analysis
   - Might need access to larger validation cohort
   - Budget for qPCR validation of top candidates

3. **Validation Strategy:**
   - Internal cross-validation first
   - External validation in year 2 with new cohort
   - Prospective validation in years 3-4

### Return to PI with Specific Plan

"Based on our discussion, I propose a three-phase approach: discovery (months 1-3), validation (months 4-12), and prospective testing (years 2-3). The first phase will give us grant-quality preliminary data, and we can assess feasibility for the larger study."

## Success Criteria

✅ **Systematic Approach:** Used iterative PI consultation to refine direction
✅ **Literature-Informed:** Identified specific gaps and opportunities  
✅ **Feasible Planning:** Broke large question into manageable phases
✅ **Resource-Aware:** Considered timeline, budget, and expertise constraints
✅ **Clinical Relevance:** Maintained focus on translational impact

## Key Takeaways

- **Always start with stakeholder input** - PIs have context you don't
- **Literature search early** - Don't reinvent the wheel
- **Iterative refinement** - Present options, get feedback, adjust
- **Plan in phases** - Minimum viable analysis → full study → validation
- **Consider resources** - Time, money, expertise all constrain what's possible