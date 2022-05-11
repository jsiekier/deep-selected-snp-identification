import os
import random
import pickle
import pickle as pkl
random.seed(3)

class Batch_creator:
    def __init__(self,input_folder,output_folder,batch_size):
        self.input_folder=input_folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        self.output_folder=output_folder
        self.batch_size=batch_size


    def create_gene_batches(self):
        all_file_names=os.listdir(self.input_folder)
        all_file_names=sorted(all_file_names)
        filter_file_names=all_file_names
        random.shuffle(filter_file_names)
        num_batches=len(filter_file_names)//self.batch_size
        rest_batch=len(filter_file_names)%self.batch_size
        batches=[]
        for i in range(num_batches):
            batches.append(filter_file_names[i*self.batch_size:(i+1)*self.batch_size])
        if rest_batch:
            batches.append(filter_file_names[self.batch_size*num_batches:])
        for i,batch in enumerate(batches):
            self.write_batch(batch,i)

    def write_batch(self, batch,batch_num):
        batch_dict={}
        for file_name in batch:
            pkl_file=open(os.path.join(self.input_folder,file_name),'rb')
            pkl_dict=pickle.load(pkl_file)
            batch_dict[file_name]=pkl_dict
            pkl_file.close()
        out_path=open(os.path.join(self.output_folder,'b_'+str(batch_num)),'wb')
        pickle.dump(batch_dict,out_path)
        out_path.close()


class Train_test_splitter:
    def __init__(self,input_folder,output_folder,folder_names,percentages):
        self.input_folder=input_folder
        self.output_folder=output_folder
        self.percentages=percentages
        self.folder_names=folder_names

    def move_files(self,names,folder_name):
        os.makedirs(os.path.join(self.output_folder,folder_name))
        for name in names:
            os.rename(os.path.join(self.input_folder,name), os.path.join(os.path.join(self.output_folder,folder_name),name))
    def make_split(self):
        all_file_names = [file_name.split("/")[-1] for file_name in os.listdir(self.input_folder)]
        all_file_names=sorted(all_file_names)
        tmp_num_files=0
        for i in range(len(self.folder_names)):
            perc_files=int(self.percentages[i]*len(all_file_names))
            file_names=all_file_names[tmp_num_files:tmp_num_files+perc_files]
            tmp_num_files+=perc_files
            self.move_files(file_names,self.folder_names[i])


