# Family-disjoint and time-out cancer-dependency transfer protocol

Protocol version: 1.4  
Locked: 2026-10-27  
Implementation namespace: `family_inductive_v2`

The 2026-10-27 implementation amendment specifies the shared-model
pipeline-correction sensitivity and the 26Q3 source receipt before external
outcome access. It also makes the overlap audit conservative for unmatched
identifier namespaces. The acquisition amendment requires an append-only
server event before downloading or hashing 26Q3 outcome bytes. These changes
follow the engineering pilot and precede the formal 45-job campaign; the pilot
remains non-scientific. The amended code and lock still require designated-server
review and tests before launch.

## Integrity boundary

This is a post-discovery development and confirmation protocol, not a
prospective preregistration. DepMap 24Q4 results and one deliberately
non-scientific pilot were inspected before this version was written. The whole
24Q4 release is therefore development evidence. No 24Q4 result may be described
as untouched confirmation.

The confirmatory same-platform dataset is DepMap 26Q3. At the initial lock,
its CRISPR outcome bytes had not been downloaded into or parsed by this project.
The method, mappings, endpoints, thresholds, 24Q4 predictions, and
temporal-analysis code must be hash-frozen before the first 26Q3 outcome access.
The freeze command refuses an existing canonical outcome, source receipt, or
access ledger. After freezing, `prepare-acquisition` creates the append-only
server ledger and binds its first event to the frozen manifest before any
download or hash of the outcome bytes. Only then may the official file be
retrieved. Before validation or parsing, a source receipt must record the
official release and file URLs, archive file ID, retrieval time, file size and
SHA-256, and a manual check against official release metadata. Freeze,
acquisition, retrieval, manual verification, and validation timestamps must
occur in that order and cannot be in the future. The receipt hash requires
reading the downloaded file after the acquisition event; the later validation
event does not represent first access to its bytes.
The receipt's `manual_official_metadata_verification` object must attest
`verified: true`, a UTC verification time, a named verifier, and the same
release URL, file URL, and archive file ID as the top-level receipt. The
receipt is provenance evidence; its self-reported URLs and attestation alone
do not prove authenticity or establish that nobody accessed the file outside
this recorded project workflow. DepMap 26Q3 is a time-out
replication, not a prospective experiment and not an independent experimental
platform.

The previously downloaded organoid screen is an orthogonal three-dimensional
assay but is not outcome-naive. It is a secondary external validation whose
prior access must be disclosed. No outcome, split, endpoint, threshold, failed
job, or baseline may be changed or omitted to obtain a positive result. All
tests, fitting, validation, aggregation, bootstrap inference, and manuscript
builds that generate evidence must run on the designated server.

## Paper mainline

The paper asks which part of cancer dependency is transferable to a gene from a
sequence-similarity component with no dependency label in training, and whether
that transfer persists in later cell models and a three-dimensional assay. The
primary estimand is the pan-context gene mean. The gene-by-cell centered
residual is secondary and cannot be inferred from pooled prediction accuracy.

The contribution is not that protein language models can predict essentiality.
That premise has direct precedents. The contribution is a leakage-controlled
study of the biological transfer boundary under simultaneous family isolation,
distance-aware reliability modeling, time-out replication, and orthogonal assay
validation. The target article type is NAR Standard Research in Computational
Biology, not NAR METHODS.

## Frozen representation and split units

The primary representation remains the residue-complete, overlap-corrected
mean embedding from `facebook/esm2_t6_8M_UR50D` at revision
`c731040fcd8d73dceaa04b0a8e6329b345b0f5df`. The exact arrays, manifests,
DepMap 24Q4 matrix, gene and cell order, and MMseqs2 component releases are
SHA-256 bound by the machine-readable lock.

The primary split unit is a connected component formed at 30% sequence identity
and 80% bidirectional coverage. Thresholds 20%, 40%, and 50% are secondary
sensitivity analyses. Complete components, rather than individual genes, are
assigned to folds. Test genes have no learned identifier, test-label fitted
feature, or test-to-train graph message.

## Primary method

The primary method is `FAMILY-DISTANCE-STACK`, an ID-free, component-cross-
fitted, distance-adaptive convex stack over three sequence-only base predictors.
For normalized query embedding `u_g`, training gene means `y_i`, and training
set `T`:

1. `G` is the mean of `y_i` over `T`.
2. `K(k,tau)` is a cosine-neighbour prediction over the top `k` training genes,
   with weights proportional to `exp((cos(u_g,u_i)-s_max)/tau)`.
3. `R(alpha)` is ridge regression on frozen ESM coordinates standardized using
   training-partition statistics only.

Let `d_g = 1 - max_i cos(u_g,u_i)`. Distances generated while cross-fitting the
training partition define `q10` and `q90`, and
`z_g = clip((d_g-q10)/(q90-q10), 0, 1)`. If `q90 == q10`, the run fails closed.
The final prediction is

`yhat_g = sum_m (a_m + b_m z_g) p_m(g)`, for `m` in `{G,K,R}`.

The constraints are `sum(a)=1`, `sum(b)=0`, nonnegative endpoint weights,
`b_K <= 0`, and `b_G >= 0`. Gate parameters minimize cross-fitted mean squared
error plus `lambda * sum(b_m^2)`. The method family never changes based on
validation performance. `STATIC-CONVEX-STACK`, obtained by fixing all `b_m=0`,
is a required ablation.

The frozen candidate grid is:

- `k`: 5, 20, 50;
- `tau`: 0.03, 0.10, 0.30;
- ridge `alpha`: 0.1, 1, 10, 100;
- gate penalty `lambda`: 0, 0.01, 0.1.

The constrained fit uses `scipy.optimize.SLSQP`, initialized at equal static
weights, with 2,000 maximum iterations, `ftol=1e-12`, and a post-fit constraint
tolerance of `1e-8`. An unsuccessful solve, non-finite objective or coefficient,
or larger constraint violation fails closed. Component folds use the first eight
bytes (big-endian) of
`SHA256(UTF8(decimal training seed) || NUL || UTF8(component identifier))`,
reduced modulo the fold count. The static stack is fitted once per `k`,
temperature and ridge-alpha triple and does not duplicate the irrelevant
`lambda` dimension. Output weight arrays contain one realized `a_m+b_m*z`
weight per query gene; the selected-baseline name is a length-one fixed-width
Unicode array. SciPy version, coefficients, solver status and constraint
residuals are retained in the audit record.

## Nested fitting and baseline selection

For each outer job, outer-test labels remain inaccessible until final scoring.
Within the specified inner-training components, deterministic three-fold
component cross-fitting produces OOF predictions for `G`, `K`, and `R` and fits
the gate. Base predictors are then refit on all inner-training genes and predict
the single registered inner-validation partition. Candidate selection maximizes
inner-validation gene-mean R-squared, then minimizes MSE, then uses the
canonical configuration string.

After selection, deterministic four-fold component cross-fitting within the
complete outer-training partition refits the gate. Base predictors are refit on
all outer-training genes and the outer-test partition is predicted exactly once.
Cross-fit fold assignment is a deterministic, component-preserving hash of the
component identifier and registered training seed.

The strongest comparator is selected separately inside every outer job using
only that job's inner validation. Ties use higher R-squared, lower MSE, then
lexicographic method name. The job writes the selected method identity and its
outer-test prediction explicitly. Campaign aggregation concatenates and
averages those per-job selected-baseline predictions. It is forbidden to pool
inner-validation scores across outer jobs to select one global method family,
because a gene that is outer-test in one job can be inner-validation in another.

Required sequence-only comparators are the global mean, uniform cosine KNN,
weighted cosine KNN, standardized ridge, static convex stack, independently
tuned ID-free mean MLP, and the matched mean-only neural ablation. The prior
mean-plus-residual neural decomposition is a secondary context model. C-PLM-MLP
and DepPLM are graph-transductive sensitivities and cannot establish the
feature-inductive primary claim.

## DepMap 24Q4 development analysis

The exact primary campaign is five outer folds by three split seeds by three
training seeds, for 45 jobs. For every split/training-seed pair, each modeled
gene appears in one outer-test fold. Every gene therefore has exactly nine OOF
predictions for each fixed method and nine predictions from the per-job selected
baseline. Consensus is the arithmetic mean over all nine appearances; no seed,
fold, failed job, or unfavorable prediction may be omitted.

The primary endpoint is full-coverage gene-mean R-squared against the consensus
per-job outer-training global predictions. The primary method effect is the
R-squared difference against the consensus per-job selected baseline, with a
paired component-block bootstrap. Pearson, Spearman, MSE, MAE, singleton versus
multi-gene components, fixed component-size bins, protein-length quartiles, and
nearest-training-distance quartiles are reported. Selective 90%, 75%, and 50%
coverage results are descriptive only and cannot replace full coverage.

Temporal and organoid validators may consume only the hash-bound
`consensus-predictions.npz` artifact. It has schema
`DepPLM-family-transfer-consensus-predictions-v2` and contains the gene and
component identities plus the primary, per-job-selected-baseline and global
consensus predictions. File and per-array hashes, dtypes and shapes are bound in
campaign `completion.json`; no training state or development outcome is exposed
through this interface.

The development signal requires positive full-coverage R-squared, a positive
component-bootstrap lower bound for MSE improvement over the global reference,
and positive R-squared in multi-gene components and the farthest distance
quartile. The method gate requires `delta R-squared >= 0.01` against the
per-job selected baseline and a component-bootstrap 95% lower bound above zero.
Failure stops the method-superiority mainline; 26Q3 cannot be used as a rescue
tuning set.

## DepMap 26Q3 time-out replication

The temporal cohort is the set of ModelIDs present in 26Q3
`CRISPRGeneEffect.csv` and absent from the exact 24Q4 CRISPR gene-effect source.
The 24Q4 exclusion set is the SHA-256-bound `data/processed/cell_lines.txt`
manifest recorded in the machine-readable lock; no alternative model list may
be substituted after outcome access.
At least 100 such models are required. Columns are parsed only when they match
`SYMBOL (ENTREZ_ID)`; normalization removes only that terminal numeric suffix,
requires unique exact symbols, and performs no synonym rescue. A gene is
eligible if it occurs in the frozen 24Q4 primary consensus and has finite 26Q3
scores in at least 80% of eligible new models. At least 5,000 genes are required.

No model or gate parameter is refit or calibrated. The target is each eligible
gene's mean over eligible new models. The primary endpoint is R-squared for the
frozen 24Q4 consensus `FAMILY-DISTANCE-STACK` prediction, using the frozen
consensus global prediction as reference. The primary comparison is delta
R-squared against the frozen consensus per-job selected baseline. Confirmation
requires positive R-squared, delta R-squared of at least 0.01, and a paired
sequence-component bootstrap 95% lower bound above zero. Spearman is a
scale-drift sensitivity result. A failed temporal gate is retained and cannot
trigger tuning, remapping, subset replacement, or threshold changes.

Because 26Q3 corrects parts of the CRISPR processing pipeline, an additional
sensitivity analysis uses models shared between 24Q4 and 26Q3; it is explicitly
not the temporal primary cohort and cannot replace it. This pre-outcome-access
analysis uses all exact ModelID matches between the two releases, with the
same exact-symbol gene mapping as the temporal primary analysis. For each gene
with at least 30 finite paired model values, the correction is the median of
26Q3 minus 24Q4 effects across those shared models. At least 5,000 genes must
qualify. The same gene-specific correction is added to each frozen 24Q4
consensus prediction and its frozen global reference, then evaluated against
the 26Q3 mean in the unchanged new-model cohort. The result is reported as a
secondary pipeline-shift sensitivity analysis; insufficient eligible genes or
any unfavorable result cannot rescue or replace the uncorrected primary gate.

The release also incorporates organoid models that may overlap the separately
downloaded three-dimensional screen. Exact ModelID overlap is audited and
reported against the 256 labels frozen before 26Q3 outcome access in
`docs/revision_v4/organoid_model_ids_predeclared.txt`. That list is the
bytewise-sorted, unique `sample_ID` column of the hash-bound organoid covariate
file. Exact string equality is the only registered overlap rule. If 26Q3 and
the organoid file use different identifier namespaces and no frozen mapping can
resolve them, overlap is reported as unresolved; it must not be reported as
zero. If at least 30 temporal models remain outside the organoid source and at
least 5,000 genes remain eligible, the same frozen analysis is repeated on that
non-overlapping subset as a required sensitivity analysis. Any non-zero or
unresolved overlap status forbids treating the temporal and organoid analyses
as statistically independent replications.

## External three-dimensional analysis

The frozen 24Q4 consensus `FAMILY-DISTANCE-STACK` and consensus per-job selected
baseline are compared with corrected organoid CRISPR log-fold changes using
per-organoid Spearman correlation over exact-symbol overlapping genes. The
absolute estimate is the arithmetic mean across eligible organoids. The paired
effect is the mean within-organoid method-minus-baseline Spearman difference.
A two-way bootstrap resamples sequence components and organoids within each
fixed tissue stratum. Raw cross-platform MSE, MAE, and R-squared are forbidden.

The external gate requires all three registered tissues, at least 100 eligible
organoids, at least 5,000 common genes per organoid, mean Spearman at least 0.20,
positive paired improvement, and a two-way bootstrap 95% lower bound above
zero. Prior access to the organoid outcome must be disclosed in the manuscript.
Exact overlap with the 26Q3 temporal cohort must also be disclosed, and
overlapping analyses cannot be combined as independent evidence.

## Secondary context endpoint

The secondary neural model predicts gene-centered cell-specific residuals.
Report residual MSE, residual R-squared against zero, and within-gene Pearson and
Spearman. A context claim requires positive residual R-squared, a positive
component-bootstrap lower bound, and mean within-gene Pearson at least 0.05.
Failure limits the manuscript to stable dependency transfer and remains a
reported result.

## Stop and reporting rules

All attempted jobs, failures, hashes, environment records, exact prediction
arrays, and negative gates remain in the release. Passing software tests is not
scientific validity or submission readiness. If the adaptive gate collapses to
static or one base method without a stable gain, the distance-adaptive novelty
claim is withdrawn. If 26Q3 or the organoid gate fails, no positive wording may
substitute for the failed endpoint.

The manuscript may be rewritten around this mainline only after server-side
campaign aggregation, independent replay, temporal validation, organoid
validation, biological interpretation, and source-bound figure/table generation
are complete. Authorship, funding, conflicts, ethics applicability, AI-use
disclosure, data licensing, public deposition, and author approval remain
separate submission blockers.
