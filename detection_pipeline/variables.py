fix_variables={
    
    "home":{
        "dr_mel":{
        "train_data":'/batches/*',#<path-to-output>/batches/*
        }
        
    },

    "dr_mel":{
        'population_pairs':[(0,1),(0,2),(0,3),(0,4),(0,5)],#population comparisons  p(0) vs p(g)
        "r":[0,0,0,0,0],# p(0)
        "o":[1,2,3,4,5],#p(g)
        'max_pool_num':6, #number of populations in the dataset
        "max_gene_len": 8000,
        'Ne': [375,417,536,491,393], # estimated Ne for every population comparison
        'Ncensus': 1000,
        'Nsampling': 1000,
        'generation': 10,
    },
}