import math
import psutil
import time
# import ray

from variables import fix_variables
import pickle
import pickle as pkl
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from Bio import SeqIO
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os
import itertools

# import wandb
"""
1. show per file the prediction distribution
2. sow manhatten plot of the prediction location
"""

num_cpus = psutil.cpu_count(logical=False)
print('-------> NUM cpus:', num_cpus)


# ray.init(num_cpus=num_cpus)

def read_polymorphisms(polymorphism_file):
    selection_stream = open(polymorphism_file, 'r')
    positions = dict()
    num_polys=0
    num_dom_freqs=0

    for line in selection_stream:
        splitted = line.replace('\n', '').split('\t')
        if splitted[0] not in positions:
            positions[splitted[0]] = dict()
        haplotype=splitted[-1].replace(' ','')
        dom_freq=haplotype.count(splitted[2])/len(haplotype)

        positions[splitted[0]][int(splitted[1])] = (splitted[2],splitted[3],dom_freq)
        if dom_freq>0.5:
            num_dom_freqs+=1
        num_polys+=1
    print('number of polymorphisms',num_polys,'percentage dom freqs',num_dom_freqs/num_polys)
    return positions


def read_cmh_values(cmh_file):
    selection_stream = open(cmh_file, 'r')
    positions = dict()
    num_cmh_values = 0
    for line in selection_stream:
        num_cmh_values += 1
        splitted = line.replace('\n', '').split('\t')
        if splitted[0] not in positions:
            positions[splitted[0]] = dict()
        if splitted[-1] != 'NaN':  # and float(splitted[-1]):
            if float(splitted[-1]):
                log_p_val = -np.log10(float(splitted[-1]))  # math.log10(float(splitted[-1]))
            else:
                log_p_val = 500  # -np.log10(1e-300)#math.log10(1e-330)

            positions[splitted[0]][int(splitted[1])] = log_p_val
        else:
            positions[splitted[0]][int(splitted[1])] = None
    print('num cmh values', num_cmh_values)
    return positions


def get_CAM_vals(all_selected_snps, ground_t=None):
    #print(ground_t,flush=True)
    all_AFC_values = []
    ground_truth = []  # num snps
    all_gene_prediction = []  # shape 5xnum snps
    # all_gene_prediction_max = []
    all_method_predictions = []  # shape 5xnum snps x num_methods
    all_gene_prediction_step1 = []  # shape 5xnum snps
    # all_gene_prediction_max_step1 = []
    ground_truth_step1 = []
    positions = []
    all_nes = []
    all_variable_positions = []
    num_variable_positions = []

    # For all replicates
    all_file_keys = []
    for c, selected_snps in enumerate(all_selected_snps):
        gene_prediction = []  # shape 5xnum snps
        # gene_prediction_max = []
        method_predictions = []
        gene_prediction_step1 = []
        Nes = []
        variable_positions = []
        # gene_prediction_max_step1 = []
        AFC_values = []
        for file_key, gene_input in selected_snps.items():


            key = file_key#.replace(del_str, '')

            if ground_t == None:
                true_selected_position = gene_input[2]
            else:
                true_selected_position = ground_t[key]

            if not len(gene_input[3]):
                Nes.append([0]*len(gene_input[1]))
            else:
                Nes.append(gene_input[3])
            variable_positions.append(gene_input[4])
            gene_prediction_step1.append(gene_input[1])

            contains_selected_pos = False
            # for all variable positions in the gene
            if not c:
                num_variable_positions.append(len(gene_input[0].items()))
            for position, ranking_vals in gene_input[0].items():
                if c == 0:
                    all_file_keys.append(key)
                    positions.append(position)
                    is_selected = False
                    for p in true_selected_position:
                        if p == position:
                            is_selected = True
                            contains_selected_pos = True
                    if is_selected:
                        ground_truth.append(1)
                    else:
                        ground_truth.append(0)
                AFC_values.append(ranking_vals[-1])
                method_predictions.append(ranking_vals[0])
                # gene_prediction_max.append(gene_pred_max)
                gene_prediction.append(gene_input[1])
            if c == 0:
                if contains_selected_pos:
                    ground_truth_step1.append(1)
                else:
                    ground_truth_step1.append(0)
        # all_file_keys.append(file_keys)
        # print(file_keys)
        all_gene_prediction.append(gene_prediction)
        # all_gene_prediction_max.append(gene_prediction_max)
        all_method_predictions.append(method_predictions)
        all_AFC_values.append(AFC_values)
        all_gene_prediction_step1.append(gene_prediction_step1)
        # all_gene_prediction_max_step1.append(gene_prediction_max_step1)
        all_nes.append(Nes)
        all_variable_positions.append(variable_positions)

    all_AFC_values = np.mean(np.asarray(all_AFC_values), axis=0)


    all_method_predictions = np.asarray(all_method_predictions)
    ground_truth = np.asarray(ground_truth)

    ground_truth_step1 = np.asarray(ground_truth_step1)
    all_gene_prediction_step1 = np.asarray(all_gene_prediction_step1)

    print('Ne shape',np.asarray( all_nes).shape)
    return ground_truth, all_gene_prediction, all_method_predictions, all_AFC_values, \
           ground_truth_step1, all_gene_prediction_step1, all_file_keys, positions, all_nes, all_variable_positions,num_variable_positions


def get_AUC_values(all_gene_prediction, all_method_predictions, prediction_threshold, min_AFC_threshold, m_idx,
                   AFC_values, summ_method, step1_preds, step1_preds_max):
    sorted_AFC_values = sorted(AFC_values, reverse=True)
    afc_idx = min(int(len(AFC_values) * min_AFC_threshold), len(AFC_values) - 1)
    min_AFC = sorted_AFC_values[afc_idx]
    sorted_all_gene_prediction = sorted(all_gene_prediction, reverse=True)
    pred_idx = min(int(len(all_gene_prediction) * prediction_threshold), len(all_gene_prediction) - 1)
    threshold = sorted_all_gene_prediction[pred_idx]

    methos_predictions = get_method_ranking(all_method_predictions, step1_preds, step1_preds_max, m_idx, summ_method)
    predicted = []
    num_zeros=0
    for m, p, a in zip(methos_predictions, all_gene_prediction, AFC_values):
        if p >= threshold and a >= min_AFC:
            predicted.append(m)#m
        else:
            predicted.append(0)
            num_zeros+=1
    print('num_zeros',num_zeros,threshold,min_AFC)
    return np.asarray(predicted)


def get_method_ranking(all_method_predictions, step1_preds, step1_preds_max, m_idx, summ_method='mean_normal'):

    methos_predictions = all_method_predictions[:, :, m_idx]
    min_value = np.min(methos_predictions)
    if min_value < 0:
        methos_predictions = methos_predictions - min_value
    methos_predictions = methos_predictions / np.max(methos_predictions)
    if 'ranking' in summ_method:
        extreme_preds = []
        for preds in methos_predictions:
            # uniform ranking:
            preds_arr = preds.tolist()
            # sort preds asccording to ranking
            sorted_preds = sorted(enumerate(preds_arr), key=lambda x: x[1])
            # sort preds acording to initial order
            preds = [(x[0] / len(preds_arr)) for x in sorted(enumerate(sorted_preds), key=lambda y: y[1])]
            extreme_preds.append(preds)
        methos_predictions = np.asarray(extreme_preds)

    if 'mean' in summ_method:
        methos_predictions = np.mean(methos_predictions, axis=0)
    elif 'maxw' in summ_method:

        multiplier = []
        for rep_num, rep_data in enumerate(step1_preds_max):
            absolute_predictions = [1 if e > 0.5 else 0 for e in rep_data]
            multiplier.append(np.mean(np.asarray(absolute_predictions)))
        multiplier = np.asarray(multiplier)[:, np.newaxis]
        methos_predictions = np.mean(methos_predictions * multiplier, axis=0)
    elif 'normw' in summ_method:

        multiplier = []
        for rep_num, rep_data in enumerate(step1_preds):
            absolute_predictions = [1 if e > 0.5 else 0 for e in rep_data]
            multiplier.append(np.mean(np.asarray(absolute_predictions)))
        multiplier = np.asarray(multiplier)[:, np.newaxis]
        methos_predictions = np.mean(methos_predictions * multiplier, axis=0)
    return methos_predictions



def get_results(ground_truth, predictions):
    average_precision_p = metrics.average_precision_score(ground_truth, predictions)
    fpr_p, tpr_p, thresholds = metrics.roc_curve(ground_truth, predictions, pos_label=1)
    roc_auc_p = metrics.auc(fpr_p, tpr_p)
    return roc_auc_p, average_precision_p


def get_ground_truth(folder_name, animal, work_from):
    ground_truth = dict()
    for file in glob.glob(fix_variables[work_from][animal][folder_name + '_data']):
        gene_file = open(file, 'rb')
        data_batch = pkl.load(gene_file)
        for gene_name, data in data_batch.items():
            positions = data['selected_pos']
            ground_truth[file.split('\\')[-1] + '\t' + gene_name] = positions
    return ground_truth


def calc_acc(all_gene_prediction_step1):
    acc_mean=0
    for rep_num, rep_data in enumerate(all_gene_prediction_step1):
        absolute_predictions = [1 if e > 0.5 else 0 for e in rep_data]
        absolute_predictions = np.asarray(absolute_predictions)
        acc=np.mean(absolute_predictions)
        print('Replicate', rep_num, 'accuracy',acc , 'mean prediction',
              np.mean(rep_data, axis=0))
        acc_mean+=acc
    if len(all_gene_prediction_step1):
        print('Mean accuracy all replicates:',acc_mean/len(all_gene_prediction_step1))


def read_selection_pos(selection_path):
    selection_stream = open(selection_path, 'r')
    positions = dict()
    num_selection_pos = 0
    for line in selection_stream:

        splitted = line.replace('\n', '').split('\t')
        if splitted[0] not in positions:
            positions[splitted[0]] = []
        positions[splitted[0]].append((int(splitted[1]), splitted[2], float(splitted[3])))
        num_selection_pos += 1
    print('num selection positions', num_selection_pos)
    selection_stream.close()
    return positions



def norm_ranking(ranking):
    ranking=np.asarray(ranking)
    spectrum=np.max(ranking)-np.min(ranking)
    if spectrum:
        return (ranking-np.min(ranking))/spectrum
    else:
        return np.zeros((ranking.shape[0]))


def combine_ranking(cmh_step1, max_pred_step1,ground_truth_step1, title):
    # compare cmh with predictions + combine results
    influence_weights = [0.0,.1,.2,.3,.4,0.5,0.6,0.7,0.8,1.0]
    cmh_step1_norm = norm_ranking(np.copy(cmh_step1))
    prediction_step1_norm = norm_ranking(max_pred_step1)

    for w in influence_weights:
        ranking = cmh_step1_norm * w + prediction_step1_norm*(1 - w)
        roc_auc_p, average_precision_p = get_results(ground_truth_step1, ranking)
        print(title, 'AUC', roc_auc_p, 'mean prec', average_precision_p, 'weight', w,
              flush=True)





def analyse_prediction2(best_predictions, ground_truth, file_keys, positions, AFC_values, work_from, animal, folder_name,
                       max_gene_len, generations, model_name, title, extendet_selection_positions, cmh_values=dict(),
                        num_variable_positions=None,ground_truth_step1=None,mean_pred=None):
    folder = fix_variables[work_from][animal][folder_name + '_data'][:-1]
    position_idx=0
    snp_infos = []
    cmh_step2 = []
    cmh_step1=[]
    max_pred_step1=[]
    none_counter=0
    for num_positions in num_variable_positions:
        key=file_keys[position_idx]
        file_name, gene_name = key.split('\t')
        gene_file = open(file_name, 'rb')
        data_batch = pkl.load(gene_file)
        data = data_batch[gene_name]
        position_cmh=[]
        position_predictions=[]
        for c in range(num_positions):
            position_predictions.append(best_predictions[position_idx])
            gt=ground_truth[position_idx]
            position=positions[position_idx]
            if data['strand'] != '-':
                tmp = position + data['start'] + 1
            else:
                tmp = data['end'] - position


            if tmp in cmh_values[data['ref']]:
                cmh_val = cmh_values[data['ref']][tmp]
                if cmh_val == None:
                    cmh_val = 0
                    none_counter+=1
                #statistics[-5].append(1.0)
                # print('right',statistics[2][-1],a)
            else:
                cmh_val = 0
                #statistics[-5].append(0.0)
                print('Error')
            position_cmh.append(cmh_val)
            if len(data['selected_pos']) and gt:
                snp_infos.append((extendet_selection_positions[data['ref']][tmp],cmh_val,best_predictions[position_idx]))
            else:
                snp_infos.append((None, cmh_val, best_predictions[position_idx]))
            position_idx+=1
        if len(position_cmh):
            cmh_step1.append(max(position_cmh))
        else:
            cmh_step1.append(0)
        cmh_step2.extend(position_cmh)
        
        if len(position_predictions):
            max_pred_step1.append(max(position_predictions))
        else:
            max_pred_step1.append(0)
        

    ranking_analysis_folder = 'models/' + model_name + '/' + title + '_ranking_analysis'
    if not os.path.exists(ranking_analysis_folder):
        os.mkdir(ranking_analysis_folder)

    combine_ranking(cmh_step1,max_pred_step1,ground_truth_step1,title='result CMH + prediction step 1:')
    combine_ranking(cmh_step2, best_predictions, ground_truth, title='result CMH + prediction step 2:')
    print('num_pos_positions',np.sum(ground_truth),'num variable positions',ground_truth.shape[0],'num cmh none positions',none_counter)





def extend_selected_positions(selection_positions, polymorphism_values):
    for chrom, position_arr in selection_positions.items():

        selection_positions[chrom] = dict()
        for position in position_arr:
            infos = polymorphism_values[chrom][position[0]]
            flip = position[1] != infos[1]
            selection_positions[chrom][position[0]] = (infos[2], flip, position[2])  # frequencie,flip,effect size

    return selection_positions


def calculate_step1_result(ground_truth_step1, all_gene_prediction_step1, all_gene_var_positions, all_nes,
                           all_variable_positions):


    all_gene_prediction_step1 = np.asarray(all_gene_prediction_step1)
    all_nes = np.asarray(all_nes)
    all_variable_positions = np.asarray(all_variable_positions)
    variable_position_mask = np.copy(all_variable_positions)
    variable_position_mask[variable_position_mask > 0] = 1  # (#replicates,#genes)
    # shape (#replicates,#genes,#cams) result shape (#genes)/(#variable positions)
    print(all_gene_prediction_step1.shape)
    res_shape=all_gene_prediction_step1.shape
    if len(res_shape)>2 and all_nes.shape[-1]>1:

        all_gene_prediction_step1=np.reshape(all_gene_prediction_step1,(res_shape[0],res_shape[1],res_shape[2]))
        # 1. calculate simple mean prediction
        mean_pred = np.mean(all_gene_prediction_step1, axis=-1) * variable_position_mask
        # 2. calculate mean of max prediction
        all_max_preds = []
        for replicate in all_gene_prediction_step1:
            replicate_preds = []
            for genes in replicate:
                gene_predictions = []
                for prediction in genes:
                    if prediction >= 0.5:
                        gene_predictions.append(prediction)
                    else:
                        gene_predictions.append(1 - prediction)
                replicate_preds.append(gene_predictions)
            all_max_preds.append(replicate_preds)
        all_max_preds = np.asarray(all_max_preds)
        max_pred = np.mean(all_max_preds, axis=-1) * variable_position_mask
        # 3. calculate variance of prediction / max (+ reverse for ranking)
        var_pred = np.std(all_gene_prediction_step1, axis=-1)
        var_pred=norm_ranking(var_pred)
        var_pred=(1-var_pred)* variable_position_mask


        var_max_pred = np.std(all_max_preds, axis=-1)
        var_max_pred=norm_ranking(var_max_pred)
        var_max_pred=(1-var_max_pred)* variable_position_mask
        # 4. calculate variance of prediction / max (+ reverse for ranking) + integrate Ne variation

        print(all_gene_prediction_step1.shape,all_nes.shape)

        ne_var_pred =  np.std(all_gene_prediction_step1, axis=-1) - np.std(all_nes, axis=-1)
        ne_var_pred = norm_ranking(ne_var_pred)
        ne_var_pred = (1-ne_var_pred) * variable_position_mask

        ne_var_max = np.std(all_max_preds, axis=-1) - np.std(all_nes, axis=-1)
        ne_var_max = norm_ranking(ne_var_max)
        ne_var_max = (1-ne_var_max)* variable_position_mask

        # 5. weight pred  max with num variable positions
        all_variable_positions[all_variable_positions <= 1] = 2
        all_variable_positions = np.log(all_variable_positions)
        max_pred_var_pos_scale = np.mean(all_max_preds, axis=-1) / all_variable_positions * variable_position_mask
        mean_pred_var_pos_scale = np.mean(all_gene_prediction_step1, axis=-1) / all_variable_positions * variable_position_mask

        all_step1_rankings = [(mean_pred, 'mean pred.'), (max_pred, 'max_pred'), (var_pred, 'std pred'),
                              (var_max_pred, 'std max'),
                              (ne_var_pred, 'ne var pred'), (ne_var_max, 'ne var max'),
                              (mean_pred_var_pos_scale, 'mean pred var pos scale'),
                              (max_pred_var_pos_scale, 'max pred var pos scale')]


    else:

        all_gene_prediction_step1=np.reshape(all_gene_prediction_step1,(res_shape[0],res_shape[1],res_shape[2]))
        # 1. calculate simple mean prediction
        mean_pred = np.mean(all_gene_prediction_step1, axis=-1) * variable_position_mask
        # 2. calculate mean of max prediction
        all_max_preds = []
        for replicate in all_gene_prediction_step1:
            replicate_preds = []
            for genes in replicate:
                gene_predictions = []
                for prediction in genes:
                    if prediction >= 0.5:
                        gene_predictions.append(prediction)
                    else:
                        gene_predictions.append(1 - prediction)
                replicate_preds.append(gene_predictions)
            all_max_preds.append(replicate_preds)
        all_max_preds = np.asarray(all_max_preds)
        max_pred = np.mean(all_max_preds, axis=-1) * variable_position_mask

        all_step1_rankings = [(mean_pred, 'mean pred.'), (max_pred, 'max_pred'),]
        
    # print AUC for every ranking:
    step1_ranking_results = []
    #print(all_step1_rankings)
    for ranking, name in all_step1_rankings:
        roc_auc_p, average_precision_p = get_results(ground_truth_step1, np.mean(ranking, axis=0))
        step1_ranking_results.append((roc_auc_p, average_precision_p, name))
    # ranking combinations:
    for subset in itertools.combinations(all_step1_rankings, 2):
        univariate_rank1 = norm_ranking(subset[0][0])
        univariate_rank2 = norm_ranking(subset[1][0])
        ranking_combination = np.mean(univariate_rank1 + univariate_rank2, axis=0)
        roc_auc_p, average_precision_p = get_results(ground_truth_step1, ranking_combination)
        step1_ranking_results.append((roc_auc_p, average_precision_p, subset[0][1] + ' + ' + subset[1][1]))
    # sort results and print:
    for entry in sorted(step1_ranking_results, key=lambda x: x[0]):
        print(entry)

    #create ranking for individual positions:
    individual_pos_ranking=[]
    for r,num_pos in zip(np.mean(mean_pred,axis=0),all_gene_var_positions):
        individual_pos_ranking.extend([r]*num_pos)
    return individual_pos_ranking,mean_pred,max_pred


def print_result(model_names,animal,work_from,cmh_file,polymorphism_file,selection_file,title):
    print(model_names,animal)
    AUC_Prec_result = []
    step1_result = []

    pop_pairs = fix_variables[animal]['population_pairs']  # [:2]
    max_gene_len = fix_variables[animal]['max_gene_len']
    summ_method = ['mean_normal', 'maxw_normal', 'normw_normal', 'mean_ranking', 'maxw_ranking', 'normw_ranking'][0]#

    

    model_types = ['train']
    model_folders = ['train']
    i = 0
    best_auc = 0
    grid_search_params=[(i,j,5,0) for i in [1,.8,.6,.4,.2] for j in [1,.8,.6,.4,.2]]

    for model_name in model_names:
        for model_type in model_types:
            for folder_name in model_folders:

                in_file_name = 'models/' + model_name + '/' + model_type + '_' + 'batches' + '_' + 'cam_results' + title + '.pkl'
                if os.path.isfile(in_file_name):
                    in_stream = open(in_file_name, 'rb')
                    data = pickle.load(in_stream)
                    in_stream.close()

                    all_replicates = []

                    ground_truth_dict = None
                    for p_idx, pop_pair in enumerate(pop_pairs):

                        key = str(pop_pair[0]) + '_' + str(pop_pair[1])
                        if key in data:
                            rep_data = data[key]  # model_type + '_' + folder_name + '_' +
                            all_replicates.append(rep_data['selected_snps'])
                            print(pop_pair)
                        if not p_idx:
                            ground_truth_dict = get_ground_truth(folder_name, animal, work_from)

                    ground_truth, all_gene_prediction, all_method_predictions, \
                    AFC_values, ground_truth_step1, all_gene_prediction_step1, \
                    file_keys, positions, all_nes, all_variable_positions,num_variable_positions = get_CAM_vals(all_replicates,
                                                                                         ground_truth_dict)
                    print('get cam vals')
                    # step 1 results:
                    individual_pos_ranking,mean_pred,max_pred=calculate_step1_result(ground_truth_step1,
                                                                                     all_gene_prediction_step1,
                                                                                     num_variable_positions, all_nes,
                                                                                    all_variable_positions)
                    print('start step 2',flush=True)
                    for prediction_threshold,min_AFC,m_idx,use_max_pred in grid_search_params:

                        predictions = get_AUC_values(individual_pos_ranking,
                                                     all_method_predictions,
                                                     prediction_threshold, min_AFC, m_idx,
                                                     AFC_values, summ_method,
                                                     mean_pred,max_pred)

                        roc_auc_p, average_precision_p = get_results(ground_truth, predictions)
                        AUC_Prec_result.append(
                            ((model_name, model_type, folder_name, prediction_threshold, min_AFC,
                              use_max_pred,
                              m_idx, 0), roc_auc_p, average_precision_p,predictions))
                        print(i, (
                        model_name, model_type, folder_name, prediction_threshold, min_AFC,
                        use_max_pred,
                        m_idx, 0), roc_auc_p, average_precision_p)
                        if roc_auc_p > best_auc:
                            best_auc = roc_auc_p
                            best_predictions = predictions
                        i += 1

                else:
                    print(in_file_name, '  not available!')
    print('Step 1 Accuracy')
    calc_acc(mean_pred)
    print('Step 1 result:')
    for entry in sorted(step1_result, key=lambda x: x[1]):
        print(entry)
    print('Step 2 result:')
    sorted_result = list(sorted(AUC_Prec_result, key=lambda x: x[1]))

    cmh_values = read_cmh_values(cmh_file)
    polymorphism_values = read_polymorphisms(polymorphism_file)
    selection_positions = read_selection_pos(selection_file)
    extendet_selection_positions = extend_selected_positions(selection_positions, polymorphism_values)

    print('position analysis:')
    for b_pred in AUC_Prec_result:#:-1
        print(b_pred[0:2])

        analyse_prediction2(b_pred[-1], ground_truth, file_keys, positions, AFC_values, work_from, animal,
                       model_folders[0],
                       max_gene_len, pop_pairs, model_name, title=title, cmh_values=cmh_values,
                       extendet_selection_positions=extendet_selection_positions,num_variable_positions=num_variable_positions,ground_truth_step1=ground_truth_step1,mean_pred=np.mean(mean_pred,axis=0))



