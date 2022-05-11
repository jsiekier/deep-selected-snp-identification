import pickle as pkl
import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO
import os
import argparse

from add_selection_position import add_selection_positions
from gene import Gene
from create_gene_batches import Batch_creator, Train_test_splitter

animal_index=2
class region_maker():
    def __init__(self,sync_path,gff_path,ref_genome_path,region_length,min_non_zero_entries,
                 min_coverage,max_coverage,min_coverage_single,max_coverage_single,filter_file,pop_idx_to_take):
        self.sync_path=sync_path
        self.gff_path=gff_path
        self.region_length=region_length
        self.min_non_zero_entries=min_non_zero_entries
        self.num_pools =len(pop_idx_to_take)
        self.min_coverage=min_coverage
        self.max_coverage=max_coverage
        self.min_coverage_single=min_coverage_single
        self.max_coverage_single=max_coverage_single
        self.base_to_idx={'a':0,'A':0,'t':1,'T':1,'c':2,'C':2,'g':3,'G':3}
        self.reverse_idx={0:1,1:0,2:3,3:2}
        self.filter_genes=self.read_filter_genes(filter_file)
        self.reverse_base={'A':'T','T':'A','C':'G','G':'C','X':'X','N':'N','a':'T','t':'A','c':'G','g':'C'}
        self.ref_genome_path=ref_genome_path
        in_seq_handle = open(ref_genome_path, 'r')
        self.ref_genome=dict()
        for value in SeqIO.parse(in_seq_handle, "fasta"):
            self.ref_genome[value.id] =  value._seq._data
        in_seq_handle.close()
        self.pop_idx_to_take=pop_idx_to_take


    def read_gff(self):
        '''

        :return: non overlapping genes!
        '''
        genes = dict()
        num_overlaps=0
        gff_file=open(self.gff_path,'r')
        tmp_gene_positions = dict()  # save positions to prevent gene overlaps of gff file
        num_genes=0
        for line in gff_file:
            if not line.startswith("#"):
                splitted = line.split("\t")
                if len(splitted)==9:
                    seqid, source, feature, start, end, score, strand, frame, attributes = splitted
                    add_gene=True
                    if feature == "gene":
                        new_gene = Gene(seqid, source, feature, start, end, score, strand, frame, attributes,gene_id=None)
                        #new_gene.end=min(new_gene.end,new_gene.start+self.region_length-1)#TODO delete this line for correct functionality this is just for a test run....
                        filtered=False
                        if seqid in self.filter_genes:
                            for e in self.filter_genes[seqid]:
                                if e[0]==new_gene.start and e[1]==new_gene.end:
                                    filtered=True
                                    break
                        if (new_gene.end - new_gene.start) < self.region_length and not filtered:
                            if seqid in tmp_gene_positions:

                                for e in tmp_gene_positions[seqid]:
                                    if (new_gene.start >= e[0] and new_gene.end <= e[1]) or (
                                                        new_gene.start <= e[0] and new_gene.end <= e[1] and new_gene.end >= e[0]) or (
                                                        new_gene.end >= e[1] and new_gene.start >= e[0] and new_gene.start <= e[1]):
                                        #add_gene = False #if this is commented - overlaps are accepted
                                        num_overlaps+=1
                                        break
                            else:
                                tmp_gene_positions[seqid] = []
                            if add_gene:
                                tmp_gene_positions[seqid].append((new_gene.start, new_gene.end))
                                num_genes+=1
                                if seqid not in genes:
                                    genes[seqid] = [new_gene]
                                else:
                                    genes[seqid].append(new_gene)
        print('num_overlaps',num_overlaps)
        print('num genes below max length',num_genes)
        gff_file.close()
        return genes

    def save_seq_regions(self,genes,tmp_storage,first_seqid,dest_dict):
        region_counter=0
        if first_seqid in genes:
            seq_genes=list(sorted(genes[first_seqid],key=lambda x:x.start))
            for i, gene in enumerate(seq_genes):
                # save regions of current gene:
                region_counter=self.save_regions(tmp_storage,gene,True,dest_dict,region_counter)

            genes[first_seqid] = []

        return genes


    def create_regions(self,dest_dict):
        if not os.path.exists(dest_dict):
            os.mkdir(dest_dict)
        if self.gff_path:
            genes=self.read_gff()
        else:
            genes=self.read_gene_fasta()
        # get alignment of genes:
        first_seqid = None
        first_line = True
        seq_id_counter=0
        tmp_storage = []
        sync_file=open(self.sync_path,'r')
        for line in sync_file:
            splitted = line.replace("\n", "").split("\t")
            seqid, position, base = splitted[:3]
            if animal_index==1:
                seqid = "scaff" + seqid.replace("Crip3.0_scaffold", "")
            alingment = splitted[3:]
            if first_line:
                first_line = False
                first_seqid = seqid

            elif first_seqid != seqid:
                print(first_seqid,seq_id_counter)
                seq_id_counter+=1
                if self.gff_path:
                    if first_seqid in self.ref_genome: #and seq_id_counter>=14:#TODO delete seq_id_counter>=14 this is just for faster execution
                        self.seq_id_bases=self.ref_genome[first_seqid]
                        genes=self.save_seq_regions(genes,tmp_storage,first_seqid,dest_dict)
                    else:
                        print(first_seqid,'not in gff file')
                        if first_seqid in genes:
                            print('but the genes files contains this with',len(genes[first_seqid]),'genes')
                else:

                    genes = self.save_seq_regions(genes, tmp_storage, first_seqid, dest_dict)


                tmp_storage=[]
                first_seqid=seqid

            # save genome data
            tmp_storage.append([int(position), base, alingment])
        if first_seqid in self.ref_genome:
            self.seq_id_bases = self.ref_genome[first_seqid]
            self.save_seq_regions(genes, tmp_storage, first_seqid, dest_dict)

    def save_regions(self,tmp_storage,gene,is_gene,dest_dict,region_counter):
        start=gene.start
        end=gene.end
        strand= gene.strand
        seqid=gene.seqid
        gene_alingment =  [np.zeros((self.region_length, 4), dtype=np.float32) for _ in range(self.num_pools)]

        nonzero_counter=0

        if self.gff_path:
            bases_gff=self.seq_id_bases[start:end]
            if strand=='-':
                bases_gff=bases_gff.reverse_complement()
        else:
            bases_gff=self.ref_genome[gene.gene_id]
            #fasta file format already use reverse complement! for reversed sequences!
        bases_gff=Seq(bases_gff)


        for entry in tmp_storage:
            tmp_position, tmp_base, tmp_aling = entry
            tmp_position=tmp_position-1

            if start <= tmp_position<end :
                idx_i = tmp_position - start
                if strand == '-':

                    idx_i = (end - start) - idx_i-1

                counter_ = 0

                for pool_num, pool in enumerate(tmp_aling):

                    if pool_num in self.pop_idx_to_take:

                        for idx_j, count in enumerate(pool.split(":")[:4]):
                            if strand=='-':
                                idx_j=self.reverse_idx[idx_j]
                            gene_alingment[counter_][idx_i, idx_j] = float(count)
                        counter_ += 1
            if tmp_position>end:
                break
        for pool_idx in range(self.num_pools):

            for idx_i in range(min(self.region_length, (end - start))):
                # investigate position:
                take_fasta_info = False
                sum_cov = np.sum(gene_alingment[pool_idx][idx_i])
                if animal_index == 0 or animal_index == 2:

                    if not (self.min_coverage <= sum_cov <= self.max_coverage):
                        take_fasta_info = True
                    else:
                        for idx_j in range(4):

                            if (not self.min_coverage_single[idx_j] <= gene_alingment[pool_idx][
                                idx_i, idx_j] <= self.max_coverage_single[idx_j]) and (
                            not gene_alingment[pool_idx][idx_i, idx_j] == 0):
                                take_fasta_info = True
                                break
                if sum_cov and not take_fasta_info:
                    nonzero_counter += 1

                    for idx_j in range(4):
                        gene_alingment[pool_idx][idx_i, idx_j] /= sum_cov

                else:
                    base = bases_gff[idx_i]
                    if not base == 'N':

                        for idx_j in range(4):
                            gene_alingment[pool_idx][idx_i, idx_j] = 0.0
                        idx_j = self.base_to_idx[base]
                        gene_alingment[pool_idx][idx_i, idx_j] = 1.0
                    else:
                        for idx_j in range(4):
                            gene_alingment[pool_idx][idx_i, idx_j] = 0.0


        if nonzero_counter > self.min_non_zero_entries * self.num_pools:

            out_file = open(dest_dict + seqid + "_" + str(region_counter), "wb")
            pkl.dump(
                {"alingment": gene_alingment, "start": start, "end": end,
                 "ref": seqid, "is_gene": is_gene,"strand":strand}, out_file, protocol=2)


            out_file.close()
            region_counter += 1

        return region_counter


    def read_filter_genes(self, filter_file):
        if filter_file=='':
            return dict()
        filter_file_stream=open(filter_file,'r')
        filter_gene_file=dict()
        for line in filter_file_stream:
            splitted=line.replace('\n','').split('\t')
            scaffold,start,end=splitted
            if scaffold not in filter_gene_file:
                filter_gene_file[scaffold]=[]
            filter_gene_file[scaffold].append((int(start),int(end)))
        return filter_gene_file

    def read_gene_fasta(self):
        genes = dict()
        num_overlaps = 0
        tmp_gene_positions = dict()  # save positions to prevent gene overlaps of gff file
        num_genes = 0
        in_seq_handle = open(self.ref_genome_path, 'r')
        for value in SeqIO.parse(in_seq_handle, "fasta"):
            descriptions=value.description.split(';')

            for entry in descriptions:
                if entry.strip().startswith('loc'):
                    location_data = entry.split('=')[1].split(':')
                    seqid = location_data[0]

                    if 'complement' in location_data[1]:
                        positions=location_data[1].replace('complement(','').replace(')','').split('..')
                        strand='-'
                    else:
                        strand = '+'
                        positions = location_data[1].split('..')
                    start = positions[0]
                    end = positions[1]
                    break

            add_gene = True
            if start!= None and end!=None and strand!=None:
                new_gene = Gene(seqid, '', 'gene', start, end, '', strand, '', None,gene_id=value.id)

                filtered = False
                if seqid in self.filter_genes:
                    for e in self.filter_genes[seqid]:
                        if e[0] == new_gene.start and e[1] == new_gene.end:
                            filtered = True
                            break
                if (new_gene.end - new_gene.start) < self.region_length and not filtered:
                    if seqid in tmp_gene_positions:

                        for e in tmp_gene_positions[seqid]:
                            if (new_gene.start >= e[0] and new_gene.end <= e[1]) or (
                                                new_gene.start <= e[0] and new_gene.end <= e[
                                            1] and new_gene.end >= e[0]) or (
                                                new_gene.end >= e[1] and new_gene.start >= e[
                                            0] and new_gene.start <= e[1]):
                                # add_gene = False #if this is commented - overlaps are accepted
                                num_overlaps += 1
                                break
                    else:
                        tmp_gene_positions[seqid] = []
                    if add_gene:
                        tmp_gene_positions[seqid].append((new_gene.start, new_gene.end))
                        num_genes += 1
                        if seqid not in genes:
                            genes[seqid] = [new_gene]
                        else:
                            genes[seqid].append(new_gene)
        print('num_overlaps', num_overlaps)
        print('num genes below max length', num_genes)
        in_seq_handle.close()
        return genes







def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--out_path', default='', type=str)
    argparser.add_argument('--gff_path', default='',type=str)
    argparser.add_argument('--sync_path', default='', type=str)
    argparser.add_argument('--fasta_path', default='', type=str)
    argparser.add_argument('--selection_file',default='',type=str)
    argparser.add_argument('--max_gene_len', default=8000, type=int)
    argparser.add_argument('--considered_populations', default=[0,1], type=int, nargs='+')

    arguments=argparser.parse_args()
    return arguments






if __name__ == '__main__':

    arguments=parse_args()
    base_path = arguments.out_path
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    gene_path = base_path + '/test/'

    region_creator=region_maker(
        gff_path=arguments.gff_path,
        sync_path=arguments.sync_path,
        ref_genome_path=arguments.fasta_path,
        region_length=arguments.max_gene_len,
        min_non_zero_entries=1,
        min_coverage=0,
        max_coverage=100000,
        min_coverage_single=[0]*len(arguments.considered_populations),
        max_coverage_single=[10000]*len(arguments.considered_populations),
        filter_file='',
        pop_idx_to_take=arguments.considered_populations
    )
    region_creator.create_regions(gene_path)

    print('all genes are written...')
    print('create train validate test split:')

    batch_path=base_path+'/batches/'
    #train_test_data=base_path+'train_test_split_b10/'
    batch_creator=Batch_creator(input_folder=gene_path,output_folder=batch_path,batch_size=25)
    batch_creator.create_gene_batches()
    #folder_names=['train','val','test']
    #folder_percentages=[0.6,0.1,0.3]
    #train_test_splitter=Train_test_splitter(batch_path,train_test_data,folder_names,folder_percentages)
    #train_test_splitter.make_split()

    add_selection_positions(sample_folder=base_path, selection_file=arguments.selection_file)

