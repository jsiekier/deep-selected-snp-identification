# deep-selected-snp-identification

Implementation of selected SNP and gene detection pipeline of the paper "Deep Unsupervised Identification of Selected Genes and SNPs in Pool-Seq Data from Evolving Populations" accepted in RECOMB-Genetics 2022.

# Running the code

## Dataset creation
.sync (with population estimates) and .fasta or .gff (with gene information) required.  
File with selected SNP information for evaluation (MimicrEE2 selection file format).  

python dataset_creation/preprocessing.py --out_path < path-to-output > (--gff_path < path-to-gff > OR --fasta_path < path-to-fasta >) --sync_path < path-to-sync-file > (--selection_file < path-to-selection-file >) --max_gene_len 8000 --considered_populations < array with index of populations in .sync you want to keep >  

Insert experimental metadata in detection_pipeline/variables.py  

## Training the model + tracing selected SNPs
  
python detection_pipeline/main_training_no_validation.py --animal < animal name in variables.py > --nn_name < model_name > --nn <'E' or 'AE' > --add_drift_filter < 0 = compare original data, 1 = compare with simulation >   
  
Saves snp data in external .pkl file of the created 'models/model_name' folder  
Add: --cmh_file <path to estimated cmh values (output of popoolation2/cmh-test.pl)>   
     --haplotype_file < path to haplotype file as used in MimicrEE2 >  
     --selection_file < path-to-selection-file >  
to the command to get a basic evaluation.

  
