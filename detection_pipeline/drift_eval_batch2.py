import json
import glob
import os
import pickle as pkl
import time

import numpy as np
from variables import fix_variables
from DNA_AE_layer import DNA_AE
from Data_loader import Eval_dataset_loader_debug
from helper import batch_simulation_creation
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics




method_names=["Grad_CAM", "CAM_resized", "guided_back",'org_grad_cam','org_grad_cam_ranking','grad_cam_ranking',
              'org_grad_cam_w','grad_cam_w','org_grad_cam_ranking_w','grad_cam_ranking_w']




def load_net(net_name,use_noise, model_type):
    model_folder = 'models/' + net_name
    model_param_path = model_folder + '/output.json'

    json_file = open(model_param_path, 'r')
    json_data = json.load(json_file)
    args = json_data  # json_data["parameters"]

    non_linearity =tf.keras.activations.relu#args['non_linearity']
    animal = args['animal']
    MAX_GENE_LENGTH = fix_variables[animal]["max_gene_len"]
    NUM_INIT_COLOR_CHANELS = args['num_init_color_chanels']
    NUM_LAYERS = args['num_layers']

    dilations = args['dilations']
    filter_size = args['filter_size']
    architecture = args['architecture']
    batchnorm_axis = args['batchnorm_axis']
    batchnorm_renorm = args['batchnorm_renorm']
    dropout = args['dropout']
    z_shortcut=[0,1,2,3,4,5,6,7]
    if 'z_shortcut' in args:
        z_shortcut=args['z_shortcut']
    print('batchnorm axis:', batchnorm_axis, 'renorm:', batchnorm_renorm)
    dna_cnn = DNA_AE(activation=non_linearity,
                     max_gene_length=MAX_GENE_LENGTH,
                     num_init_color_chanels=NUM_INIT_COLOR_CHANELS,
                     num_layers=NUM_LAYERS,
                     dilations=dilations,
                     filter_size=filter_size,
                     architecture=architecture,
                     batchnorm_axis=batchnorm_axis,
                     batchnorm_renorm=batchnorm_renorm,
                     dropout=dropout,
                     add_drift_filter=args['add_drift_filter'],
                     subtract_output=args['subtract_output'],
                     gauss_lay=args['gauss_lay'],
                     gauss_std=args['gauss_std'],
                     use_noise=use_noise,opt='adam',learning_rate=0.001,z_shortcut=z_shortcut)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=dna_cnn.optimizer, net=dna_cnn)
    if model_type=='latest':
        checkpoint_end = '/tf_ckpts_last'
    elif model_type=='train':
        checkpoint_end = '/tf_ckpts_train'
    elif model_type=='val':
        checkpoint_end = '/tf_ckpts'
    elif model_type=='balance':
        checkpoint_end='/tf_ckpts_balance'
    else:
        print('ERROR!!! model type not valid')

    manager = tf.train.CheckpointManager(ckpt, model_folder + checkpoint_end, max_to_keep=3)  # _train
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("ERROR: No Checkpoint file available!",model_folder  + checkpoint_end)
    return animal, MAX_GENE_LENGTH, dna_cnn,args['round_decimal']


def get_cam_results(real, other, dna_cnn, res,use_batch=False):
    if dna_cnn.subtract_output:
        if use_batch:
             #start_calc_time=time.time()
             result_0, prop, heat_resized_0, guided_backprop_0, org_grad_cam,\
             org_grad_cam_ranking,result_ranking,org_grad_cam_w,result_w,org_grad_cam_ranking_lay,org_cam_ranking_w,cam_ranking_w = dna_cnn.get_cam_subtract_batch(real,other)#,net_time,cam_time,cam_org

             res = SNP_positiones_percentage_subtract(org_grad_cam_ranking, 'org_grad_cam_ranking', res)
             res = SNP_positiones_percentage_subtract(org_grad_cam_w, 'org_grad_cam_w', res)

             res = SNP_positiones_percentage_subtract(result_ranking, 'grad_cam_ranking', res)
             res = SNP_positiones_percentage_subtract(cam_ranking_w, 'grad_cam_ranking_w', res)
             res = SNP_positiones_percentage_subtract(org_cam_ranking_w, 'org_grad_cam_ranking_w', res)
             res = SNP_positiones_percentage_subtract(result_w, 'grad_cam_w', res)

        else:
            _,result_0, prop, heat_resized_0, guided_backprop_0,org_grad_cam=dna_cnn.get_cam_subtract(real, other)
        res = SNP_positiones_percentage_subtract(result_0, "Grad_CAM", res)
        res = SNP_positiones_percentage_subtract(heat_resized_0, "CAM_resized", res)
        res = SNP_positiones_percentage_subtract(guided_backprop_0, "guided_back", res)
        res = SNP_positiones_percentage_subtract(org_grad_cam, 'org_grad_cam', res)


        return [res,prop.numpy().tolist()]
    else:
        heat_0, heat_1, result_0, result_1, prop, heat_resized_0, heat_resized_1, \
        guided_backprop_0, guided_backprop_1 = dna_cnn.get_cam(real, other)

        if np.argmax(prop):

            res = SNP_positiones_percentage(result_1, "Grad_CAM", res, dna_cnn.max_gene_length)
        else:
            #

            res = SNP_positiones_percentage(result_0, "Grad_CAM", res, dna_cnn.max_gene_length)

        res = SNP_positiones_percentage(np.abs(result_1) + np.abs(result_0), "Grad_CAM_add_abs_no_w", res,
                                        dna_cnn.max_gene_length)
        res = SNP_positiones_percentage(np.abs(heat_resized_0) + np.abs(heat_resized_1), "CAM_resized", res,
                                        dna_cnn.max_gene_length)
        res = SNP_positiones_percentage(np.abs(guided_backprop_0) + np.abs(guided_backprop_1), "guided_back", res,
                                        dna_cnn.max_gene_length)
        return [res,float(prop[0])]


def SNP_positiones_percentage(heat_map, method, result, max_gene_len):
    positions = result.keys()
    method_idx = method_names.index(method)

    for i in positions:
        heat_0 = heat_map[i]
        heat_1 = heat_map[i + max_gene_len]
        result[i][0][method_idx] = max(heat_0, heat_1)
    return result


def SNP_positiones_percentage_subtract(heat_map, method, result, use_list=False):
    positions = result.keys()
    method_idx = method_names.index(method)

    for i in positions:
        heat_0 = heat_map[i]
        if use_list:
            result[i][0][method_idx] = heat_0.tolist()
        else:
            result[i][0][method_idx] = float(heat_0)
    return result

def get_naive_snps(input_real_gene, input_other_gene, len_methods,variable_positions):
    position_dict=dict()
    for i in variable_positions:
        max_AFC = 0
        for j in range(4):
            if input_other_gene[i, j] != input_real_gene[i, j]:

                AFC = np.abs(input_other_gene[i, j] - input_real_gene[i, j])
                #if (AFC >= allele_diff):
                max_AFC = max(max_AFC, AFC)
        #if max_AFC:
        position_dict[i] = [[0] * len_methods,max_AFC]
    return position_dict


def get_variable_positions(folder,pop_pairs):
    result=dict()
    for file in sorted(glob.glob(folder)):
        gene_file = open(file, 'rb')
        data_batch = pkl.load(gene_file)
        for gene_name, data in data_batch.items():
            variable_positions=set()
            gene = data["alingment"]
            for pop_pair in pop_pairs:
                real = gene[pop_pair[0]]
                other = gene[pop_pair[1]]
                diff=(real-other)
                positions=np.where(diff!=0)
                tmp=set(positions[0].tolist())
                variable_positions=variable_positions.union(tmp)
            result[file+'_'+gene_name]=list(sorted(list(variable_positions)))
    return result





def create_complete_cam_result(work_from, model_name,num_cams,title):
    animal, _, _, _ = load_net(model_name, use_noise=False, model_type='latest')
    pop_pairs = fix_variables[animal]['population_pairs']

    Nes=fix_variables[animal]['Ne']
    Ncensus=fix_variables[animal]['Ncensus']
    Nsample=fix_variables[animal]['Nsampling']
    generation=fix_variables[animal]['generation']




    for folder in [fix_variables[work_from][animal]['train_data']]:
        folder_name = folder.split('/')[-2]
        gene_positions=get_variable_positions(folder,pop_pairs)
        for model_type in ['train']:  # ,'balance','latest',,'val'
            #'''
            in_file_name='models/' + model_name + '/' + model_type + '_' + folder_name + '_' + 'cam_results'+title+'.pkl'
            if os.path.isfile(in_file_name):
                in_stream = open(in_file_name,
                                 'rb')  # open('models/' + model_name + '/' + 'cam_results'+title+'.pkl', 'rb')
                tmp_data = pkl.load(in_stream)
                in_stream.close()
            else:
            #'''
            	tmp_data=dict()
            all_results = dict()
            for pop_pair in pop_pairs:
                #tmp_data_key=model_type + '_' + folder_name + '_' + str(pop_pair[0]) + '_' + str(pop_pair[1])
                tmp_data_key =  str(pop_pair[0]) + '_' + str(pop_pair[1])
                if tmp_data_key in tmp_data:
                    result_dict=tmp_data[tmp_data_key]
                    all_results[str(pop_pair[0]) + '_' + str(pop_pair[1])] = result_dict
                    print('save',flush=True)
            for idx,pop_pair in enumerate(pop_pairs):
                tmp_data_key = str(pop_pair[0]) + '_' + str(pop_pair[1])
                if tmp_data_key not in all_results:
                    Ne=Nes[idx]
                    animal, MAX_GENE_LENGTH, dna_cnn,decimal_round = load_net(model_name,use_noise=False, model_type=model_type)
                    print(model_type,folder_name,pop_pair)
                    result_dict = dict()
                    result_dict['methods']=method_names
                    result_dict['selected_snps']=dict()
                    folders=list(glob.glob(folder))
                    for file_idx,file in enumerate(sorted(folders)):


                        gene_file = open(file, 'rb')
                        data_batch = pkl.load(gene_file)

                        for gene_idx,(gene_name, data) in enumerate(data_batch.items()):

                            gene = data["alingment"]

                            real,_,other,Nes_new,num_variable_positions = batch_simulation_creation(gene[pop_pair[0]], gene[pop_pair[1]], Ne , Ncensus,
                                                                                  Nsample, generation, num_cams[0])


                            real = np.around(real, decimals=decimal_round)
                            other = np.around(other, decimals=decimal_round)
                            use_batch=True if num_cams else False
                            position_dict = get_naive_snps(gene[pop_pair[0]],gene[pop_pair[1]],len(method_names),gene_positions[file+'_'+gene_name])
                            result_list= get_cam_results(real,other, dna_cnn, position_dict,use_batch=use_batch)#,time_cam_calc,time_save,net_time,cam_time,cam_org

                            result_dict['selected_snps'][file+'\t'+gene_name] = result_list+[data['selected_pos'],Nes_new,num_variable_positions]



                    all_results[ str(pop_pair[0]) + '_' + str(pop_pair[1])]=result_dict
                out_stream = open('models/' + model_name + '/' +model_type+'_'+folder_name+'_'+'cam_results'+title+'.pkl', 'wb')
                pkl.dump(all_results, out_stream)
                out_stream.close()



def execute_cam(model_name,work_from,add_drift_filter):
    if add_drift_filter:
        num_cams=[(100,40)]
        title='batch100_ne40'
    else:
        num_cams=[(1,0)]
        title='batch1_ne0'
    create_complete_cam_result(work_from, model_name,num_cams,title=title)
    return title

