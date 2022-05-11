import tensorflow as tf
import glob
import numpy as np

import random
import pickle




rng = np.random.default_rng(seed=0)





def create_batch_biallelic_freqs(g0):
    biallelic_freqs, positions_major,positions_minor,del_positions=[],[],[],[]
    tmp=np.logical_and((g0!=1),(g0!=0))
    positions_tmp=np.where(tmp)
    major_freqs=[]

    positions_tmp=sorted(list(set(zip(positions_tmp[0],positions_tmp[1]))))
    #search position with highest and second highest as major and minor + recalculate freq?!
    for pos in positions_tmp:
        indices = np.argsort(g0[pos[0],pos[1]])
        if g0[pos[0],pos[1],indices[1]]:
            new_freq_major=float(g0[pos[0],pos[1],indices[3]]/(g0[pos[0],pos[1],indices[2]]+g0[pos[0],pos[1],indices[3]]))
            del_positions.append([pos[0],pos[1],indices[1]])
            if g0[pos[0],pos[1],indices[0]]:
                del_positions.append([pos[0],pos[1], indices[0]])
        else:
            new_freq_major=float(g0[pos[0],pos[1],indices[3]])
        #new_freq_major=round(new_freq_major,3)
        major_freqs.append(new_freq_major)
        biallelic_freqs.append(new_freq_major)#+=str(new_freq_major)+':'
        positions_major.append(indices[3])
        positions_minor.append(indices[2])
    batch_pos,row_pos=zip(*positions_tmp)
    return biallelic_freqs,positions_major,positions_minor,del_positions,row_pos,batch_pos,np.asarray(major_freqs)



def simulate_drift_batch(biallelic_freqs, generation,Ne,Ncensus,Nsampling):
    Ne*=2 # diploid org.
    Ne=max(Ne,10)
    new_freqs=np.asarray(biallelic_freqs)
    q=1.0-new_freqs
    n=len(biallelic_freqs)
    for _ in range(generation):

        new_freqs=rng.binomial(Ne,new_freqs,n)/Ne
    if Nsampling<Ncensus:
        new_freqs=rng.hypergeometric(ngood=new_freqs*Ncensus*2,nbad=q*Ncensus*2,nsample=Nsampling*2,size=n)

    return new_freqs




def batch_simulation_creation(g0,gn,Ne,Ncensus,Nsampling,generation):

    biallelic_freqs, positions_major, positions_minor, del_positions,rows,batch_positions,major_freqs=create_batch_biallelic_freqs(g0)
    if len(positions_major):
        gN_freqs=simulate_drift_batch(biallelic_freqs,generation,Ne,Ncensus,Nsampling)
        gN=np.copy(g0)
        if len(del_positions):
            batch,rows_,cols=zip(*del_positions)
            gN[batch,rows_,cols]=[0]*len(del_positions)
            gn[batch, rows_, cols] = [0] * len(del_positions)
        new_g0=np.copy(gN)
        gN[batch_positions,rows, positions_major] = gN_freqs
        gN_minor_freqs=1-gN_freqs
        gN[batch_positions,rows, positions_minor] = gN_minor_freqs

        new_g0[batch_positions,rows, positions_major] = major_freqs
        g0_minor_freqs=1-major_freqs
        new_g0[batch_positions,rows, positions_minor] = g0_minor_freqs

        #fake_g_,new_g0_,new_gn_
        return gN,new_g0,gn

    return g0,g0








def get_batch_simulations(batch_keys,g0_idx,gn_idx, Ne, Ncensus, Nsampling, generation, r_path,decimals):
    genes0, genesN = [], []
    for i, key in enumerate(batch_keys):
        key = str(key).replace("b'", '').replace("'", '').split('\\t')
        # print(key)
        filename, gene_name = key[0], key[1]
        gene_file = open(filename, "rb")
        data_batch = pickle.load(gene_file)
        data = data_batch[gene_name]
        alingment = data["alingment"]
        genes0.append(alingment[g0_idx])
        genesN.append(alingment[gn_idx])

    genes0=np.asarray(genes0)
    genesN=np.asarray(genesN)
    fake_g_, new_g0_, new_gn_ = batch_simulation_creation(g0=genes0, gn=genesN, Ne=Ne, Ncensus=Ncensus,
                                                          Nsampling=Nsampling, generation=generation,
                                                          r_path=r_path)
    genes0 = np.around(genes0, decimals=decimals)  # shape: #genes#pool_num#gene_len#nukleotides
    genesN = np.around(genesN, decimals=decimals)
    fake_g_ = np.around(fake_g_, decimals=decimals)
    return genes0,genesN,fake_g_


class Eval_dataset_loader(tf.data.Dataset):

    def _generator(keys, population_pair, Ne, Ncensus, Nsampling, generation, r_path,decimals):
        batchsize=50
        g0_idx,gn_idx=population_pair
        num_batches=len(keys)//batchsize
        rest=len(keys)%batchsize
        adaptation_labels = np.zeros((1), dtype=np.float32)
        adaptation_labels[0] = 1
        for batch_num in range(num_batches):
            batch_keys=keys[batch_num*batchsize:(batch_num+1)*batchsize]
            #print(batch_keys)
            genes0, genesN, fake_g_=get_batch_simulations(batch_keys,g0_idx,gn_idx, Ne, Ncensus, Nsampling, generation, r_path,decimals)
            for g0,gN,gN_fake in zip(genes0,genesN,fake_g_):
                yield( gN, g0, gN_fake,g0, adaptation_labels)
        if rest:
            batch_keys = keys[num_batches * batchsize:]
            genes0, genesN, fake_g_ = get_batch_simulations(batch_keys, g0_idx, gn_idx, Ne, Ncensus, Nsampling,
                                                            generation, r_path, decimals)
            for g0, gN, gN_fake in zip(genes0, genesN, fake_g_):
                yield (gN, g0, gN_fake, g0, adaptation_labels)


    def __new__(self, keys, population_pair, max_gene_len, num_init_color_channels, Ne, Ncensus,
                Nsampling, generation, r_path, decimals):


        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=(
            tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32),
            output_shapes=((max_gene_len, num_init_color_channels), (max_gene_len, num_init_color_channels),
                           (max_gene_len, num_init_color_channels), (max_gene_len, num_init_color_channels), (1)),
            args=(keys, population_pair, Ne, Ncensus, Nsampling, generation, r_path,decimals)
        )


def make_instances(g0_idx,gn_idx,file_names, Ne, Ncensus, Nsampling, generation, r_path, decimals):
    genes0=[]
    genesN=[]
    selection_info=[]
    for i, filename in enumerate(file_names):
        gene_file = open(filename, "rb")
        data_batch = pickle.load(gene_file)
        for gene_name, data in data_batch.items():
            alingment = data["alingment"]
            genes0.append(alingment[g0_idx])
            genesN.append(alingment[gn_idx])
            if data['selected_pos']:
                selection_info.append(1)
            else:
                selection_info.append(0)
    # genes=np.around(np.asarray(genes),decimals=decimals)# shape: #genes#pool_num#gene_len#nukleotides
    g0=np.asarray(genes0)
    gn=np.asarray(genesN)


    fake_g_, new_g0_, new_gn_ = batch_simulation_creation(g0=g0, gn=gn, Ne=Ne, Ncensus=Ncensus,
                                                          Nsampling=Nsampling, generation=generation, r_path=r_path)

    new_g0 = np.around(new_g0_, decimals=decimals)
    new_gn = np.around(new_gn_, decimals=decimals)
    fake_g = np.around(fake_g_, decimals=decimals)

    return new_g0,new_gn,fake_g,selection_info


class Eval_dataset_loader_debug(tf.data.Dataset):

    def _generator(population_pair,file_names ,Ne, Ncensus, Nsampling, generation, r_path,decimals):
        #print(population_pair,file_names ,Ne, Ncensus, Nsampling, generation, r_path,decimals)
        batchsize=5
        g0_idx,gn_idx=population_pair
        num_batches=len(file_names)//batchsize
        rest=len(file_names)%batchsize

        adaptation_labels = np.zeros((1), dtype=np.float32)
        selection_label = np.zeros((1), dtype=np.int8)
        adaptation_labels[0] = 1
        for batch_num in range(num_batches):
            new_g0, new_gn, fake_g,selection_infos=make_instances(g0_idx,gn_idx,file_names[batch_num*batchsize:(batch_num+1)*batchsize],
                                                  Ne,Ncensus,Nsampling,generation,r_path,decimals)


            for g0,gN, gN_fake,selection_inf in zip(new_g0, new_gn, fake_g,selection_infos):
                selection_label[0]=selection_inf
                yield( gN, g0, gN_fake,g0, adaptation_labels,selection_label)
        if rest:
            new_g0, new_gn, fake_g, selection_infos = make_instances(g0_idx, gn_idx, file_names[num_batches*batchsize:],
                                                                     Ne, Ncensus, Nsampling, generation, r_path,
                                                                     decimals)

            for g0, gN, gN_fake, selection_inf in zip(new_g0, new_gn, fake_g, selection_infos):
                selection_label[0] = selection_inf
                yield (gN, g0, gN_fake, g0, adaptation_labels,selection_label)


    def __new__(self, gene_dictionary, population_pair, max_gene_len, num_init_color_channels, Ne, Ncensus,
                Nsampling, generation, r_path, decimals):
        file_names=[]
        if gene_dictionary != "":
            for filename in glob.glob(gene_dictionary):
                file_names.append(filename)
        file_names=list(sorted(file_names))


        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=(
            tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.int8),
            output_shapes=((max_gene_len, num_init_color_channels), (max_gene_len, num_init_color_channels),
                           (max_gene_len, num_init_color_channels), (max_gene_len, num_init_color_channels), (1),(1)),
            args=( population_pair, file_names,Ne, Ncensus, Nsampling, generation, r_path,decimals)
        )


class ArtificialDataset_debug(tf.data.Dataset):

    def _generator(num_samples,num_gene_batches,file_names,pair_idx,population_pairs,decimals):
        file_names = random.sample(set(file_names), num_gene_batches)
        genes = []
        selection_info=[]
        for i,filename in enumerate(file_names):
            gene_file = open(filename, "rb")
            data_batch = pickle.load(gene_file)
            for gene_name, data in data_batch.items():
                alingment = data["alingment"]
                genes.append(alingment)
                if data['selected_pos']:
                    selection_info.append(1)
                else:
                    selection_info.append(0)

        gene_ids = list(range(len(genes)))

        for sample_idx in range(num_samples): # for number of repeat iterations:
            random.shuffle(gene_ids)

            for g_id in gene_ids:
                gene=genes[g_id]
                pair_number = random.choice(pair_idx)
                adaptation_lab=random.choice([0,1])
                adaptation_labels = np.zeros((1), dtype=np.float32)
                adaptation_labels[0]=adaptation_lab
                real=np.around(gene[population_pairs[pair_number][adaptation_lab]], decimals=decimals)
                other=np.around(gene[population_pairs[pair_number][(adaptation_lab+1)%2]], decimals=decimals)
                selection_labels = np.zeros((1), dtype=np.uint8)
                selection_labels[0] = selection_info[g_id]

                yield (real,other,adaptation_labels,selection_labels)


    def __new__(self, gene_dictionary,population_pairs,max_gene_len,num_init_color_channels,decimals,seed=3,num_gene_batches=3,num_samples=3,):
        np.random.seed(seed)
        random.seed(seed)
        file_names = []
        if gene_dictionary != "":
            for filename in glob.glob(gene_dictionary):
                file_names.append(filename)

        num_population_pairs = list(range(len(population_pairs)))
        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=(tf.dtypes.float32,tf.dtypes.float32,tf.dtypes.float32,np.uint8),
            output_shapes=((max_gene_len,num_init_color_channels),(max_gene_len,num_init_color_channels),(1),(1)),
            args=(num_samples,num_gene_batches,file_names,num_population_pairs,population_pairs,decimals)
        )


class Dataset_loader_debug(tf.data.Dataset):

    def _generator(num_samples,num_gene_batches,file_names,population_pairs,Ne,Ncensus,Nsampling,generation,decimals):
        genes = []
        selection_info=[]
        num_replicates = list(range(len(population_pairs)))

        file_names = random.sample(set(file_names), num_gene_batches)
        for i,filename in enumerate(file_names):
            gene_file = open(filename, "rb")
            data_batch = pickle.load(gene_file)
            for gene_name, data in data_batch.items():
                alingment = data["alingment"]
                genes.append(alingment)
                if data['selected_pos']:
                    selection_info.append(1)
                else:

                    selection_info.append(0)
        #genes=np.around(np.asarray(genes),decimals=decimals)# shape: #genes#pool_num#gene_len#nukleotides
        g=np.transpose(genes,(1,0,2,3))

        new_g0,new_gn,fake_g=[],[],[]
        for c,(g0_idx,gn_idx) in enumerate(population_pairs):
            g0=g[g0_idx]
            gn=g[gn_idx]
            curr_Ne=rng.normal(Ne[c],40,1)[0]
            fake_g_,new_g0_,new_gn_ = batch_simulation_creation(g0=g0,gn=gn,Ne=curr_Ne,Ncensus=Ncensus,
                                                                Nsampling=Nsampling,generation=generation)
            new_g0.append(new_g0_)
            new_gn.append(new_gn_)
            fake_g.append(fake_g_)

        new_g0=np.around(np.transpose(np.asarray(new_g0),(1,0,2,3)),decimals=decimals)
        new_gn =np.around(np.transpose( np.asarray(new_gn),(1,0,2,3)),decimals=decimals)
        fake_g = np.around(np.transpose(np.asarray(fake_g),(1,0,2,3)),decimals=decimals)#np.copy(new_g0)#

        gene_ids=list(range(len(genes)))
        five_lab = np.zeros((1), dtype=np.float32)
        five_lab[0]=0.5
        for sample_idx in range(num_samples): # for number of repeat iterations:
            random.shuffle(gene_ids)

            for g_id in gene_ids:
                pair_number = random.choice(num_replicates)
                adaptation_lab=random.choice([0,1])
                fake_lab = random.choice([0, 1])
                adaptation_labels = np.zeros((1), dtype=np.float32)
                fake_labels = np.zeros((1), dtype=np.float32)
                selection_labels = np.zeros((1), dtype=np.uint8)
                selection_labels[0]=selection_info[g_id]
                adaptation_labels[0] = adaptation_lab
                fake_labels[0]=fake_lab
                if adaptation_lab:
                    real = new_gn[g_id, pair_number]
                    other = new_g0[g_id, pair_number]
                    fake_real = fake_g[g_id, pair_number]
                    fake_other = new_g0[g_id, pair_number]
                else:
                    real = new_g0[g_id, pair_number]
                    other = new_gn[g_id, pair_number]
                    fake_real = new_g0[g_id, pair_number]
                    fake_other = fake_g[g_id, pair_number]
                if fake_lab:
                    fake_inst=fake_g[g_id, pair_number]
                    real_inst=new_gn[g_id, pair_number]
                else:
                    fake_inst = new_gn[g_id, pair_number]
                    real_inst = fake_g[g_id, pair_number]

                yield (real,other,fake_real,fake_other,adaptation_labels,selection_labels,fake_inst,real_inst,fake_labels,five_lab)


    def __new__(self, gene_dictionary,population_pairs,max_gene_len,num_init_color_channels,Ne,Ncensus,
                Nsampling,generation,decimals,seed=3,num_gene_batches=3,num_samples=3):
        np.random.seed(seed)
        random.seed(seed)
        file_names = []

        if gene_dictionary != "":
            for filename in glob.glob(gene_dictionary):
                file_names.append(filename)

        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=(tf.dtypes.float32,tf.dtypes.float32,tf.dtypes.float32,tf.dtypes.float32,tf.dtypes.float32,tf.dtypes.float32,tf.dtypes.float32,tf.dtypes.float32,tf.dtypes.float32,tf.dtypes.float32),
            output_shapes=((max_gene_len,num_init_color_channels),(max_gene_len,num_init_color_channels),(max_gene_len,num_init_color_channels),(max_gene_len,num_init_color_channels),(1),(1),(max_gene_len,num_init_color_channels),(max_gene_len,num_init_color_channels),(1),(1)),
            args=(num_samples,num_gene_batches,file_names,population_pairs,Ne,Ncensus,Nsampling,generation,decimals)
        )


class Dataset_loader_pretrain(tf.data.Dataset):

    def _generator(num_samples,num_gene_batches,file_names,population_pairs,Ne,Ncensus,Nsampling,generation,r_path,decimals):
        genes = []
        selection_info=[]
        num_replicates = list(range(len(population_pairs)))

        file_names = random.sample(set(file_names), num_gene_batches)
        for i,filename in enumerate(file_names):
            gene_file = open(filename, "rb")
            data_batch = pickle.load(gene_file)
            for gene_name, data in data_batch.items():
                alingment = data["alingment"]
                genes.append(alingment)
                if data['selected_pos']:
                    selection_info.append(1)
                else:

                    selection_info.append(0)
        #genes=np.around(np.asarray(genes),decimals=decimals)# shape: #genes#pool_num#gene_len#nukleotides
        g=np.transpose(genes,(1,0,2,3))

        new_g0,new_gn,fake_g=[],[],[]
        for c,(g0_idx,gn_idx) in enumerate(population_pairs):
            g0=g[g0_idx]
            gn=g[gn_idx]
            curr_Ne=rng.normal(Ne[c],80,1)[0]
            fake_g_,new_g0_,new_gn_ = batch_simulation_creation(g0=g0,gn=gn,Ne=curr_Ne,Ncensus=Ncensus,
                                                                Nsampling=Nsampling,generation=generation,r_path=r_path)
            #new_g0.append(new_g0_)
            new_gn.append(new_gn_)
            fake_g.append(fake_g_)

        #new_g0=np.around(np.transpose(np.asarray(new_g0),(1,0,2,3)),decimals=decimals)
        new_gn =np.around(np.transpose( np.asarray(new_gn),(1,0,2,3)),decimals=decimals)
        fake_g = np.around(np.transpose(np.asarray(fake_g),(1,0,2,3)),decimals=decimals)

        gene_ids=list(range(len(genes)))

        for sample_idx in range(num_samples): # for number of repeat iterations:
            random.shuffle(gene_ids)

            for g_id in gene_ids:

                fake_labels = np.zeros((1), dtype=np.float32)
                pair_number = random.choice(num_replicates)

                if fake_labels:
                    fake_inst=fake_g[g_id, pair_number]

                else:
                    fake_inst = new_gn[g_id, pair_number]

                yield (fake_inst)


    def __new__(self, gene_dictionary,population_pairs,max_gene_len,num_init_color_channels,Ne,Ncensus,
                Nsampling,generation,r_path,decimals,seed=3,num_gene_batches=3,num_samples=3):
        np.random.seed(seed)
        random.seed(seed)
        file_names = []

        if gene_dictionary != "":
            for filename in glob.glob(gene_dictionary):
                file_names.append(filename)

        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=(tf.dtypes.float32),
            output_shapes=((max_gene_len,num_init_color_channels)),
            args=(num_samples,num_gene_batches,file_names,population_pairs,Ne,Ncensus,Nsampling,generation,r_path,decimals)
        )

