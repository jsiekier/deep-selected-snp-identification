# deep-selected-SNP-identification

Implementation of selected SNP and gene detection pipeline of the paper "Deep Unsupervised Identification of Selected Genes and SNPs in Pool-Seq Data from Evolving Populations" accepted in RECOMB-Genetics 2022.

# Running the code

## Dataset creation
.sync (with population estimates) and .fasta or .gff (with gene information) required.
File with selected SNP information for evaluation (MimicrEE2 selection file format).

python dataset_creation/preprocessing.py --out_path <path-to-output> (--gff_path <path-to-gff> OR --fasta_path <path-to-fasta>) --sync_path <path-to-sync-file> (--selection_file <path-to-selection-file>) --max_gene_len 8000 --considered_populations <array with index of populations in .sync you want to keep>


## Training the model

## Tracing selected SNPs

## Evaluation
