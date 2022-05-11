import glob
import pickle as pkl

def read_selection_pos(selection_path):
    selection_stream=open(selection_path,'r')
    positions=dict()
    for line in  selection_stream:
        splitted=line.replace('\n','').split('\t')
        if splitted[0] not in positions:
            positions[splitted[0]]=[]
        positions[splitted[0]].append(int(splitted[1]))
    selection_stream.close()
    return positions

def add_selection_positions(sample_folder,selection_file):
    if selection_file!='':
        selection_positions = read_selection_pos(selection_file)
    else:
        selection_positions=dict()
    for folder_name in ['/batches']:
        for c, file in enumerate(glob.glob(sample_folder+folder_name + '/*')):
            gene_file = open(file, 'rb')
            data_batch = pkl.load(gene_file)
            new_data_batch=dict()
            for gene_name, data in data_batch.items():
                start = data['start']
                end = data['end']
                ref = data['ref']
                positions = []
                if ref in selection_positions:

                    for position in selection_positions[ref]:

                        if start <= position - 1 < end:
                            tmp_position=position-1-start

                            if data['strand']=='-':
                                tmp_position= (end - start) - tmp_position-1

                            positions.append(tmp_position)
                            break
                data['selected_pos']=positions
                new_data_batch[gene_name]=data
            gene_file.close()
            gene_file=open(file, 'wb')
            pkl.dump(new_data_batch,gene_file)
            gene_file.close()

