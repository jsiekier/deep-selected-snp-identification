from Data_loader import Dataset_loader_debug,ArtificialDataset_debug,Dataset_loader_pretrain
from DNA_AE_layer import DNA_AE
import argparse
import tensorflow as tf
from variables import fix_variables
import json

import os
from drift_eval_batch2 import execute_cam
from plot_selection_result import print_result
metric_names=['align discrimination entropy', 'align accuracy', 'fake entropy',
                               'fake accuracy', 'average selection precision', 'average selection precision max',
                               'selection AUC', 'selection AUC max','drift distinguish acc']
pretrain_metrics=['loss','entropy','kld','l1_reinforce']
num_metrics=len(metric_names)



def log_pretrain(statistics,step):

    result_dict = dict()
    for name,val in zip(pretrain_metrics,statistics):
        result_dict[name] = float(val)
        print(name,round(float(val),5),end='\t')
    print(flush=True)
    result_dict["Step"] = step


def parse_args():
    
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--non_linearity', default='relu', type=str)
    argparser.add_argument('--batch_size', default=100, type=int)
    argparser.add_argument('--num_gene_batches', default=2, type=int)#10
    argparser.add_argument('--work_from', default="home", type=str)
    argparser.add_argument('--animal', default='dr_mel', type=str)
    argparser.add_argument('--num_init_color_chanels', default=4, type=int)
    argparser.add_argument('--num_layers', default=6, type=int)
    argparser.add_argument('--train_sessions', default=8000, type=int)#10000
    argparser.add_argument('--repeat_iterations', default=2, type=int)#4
    argparser.add_argument('--print_every', default=200, type=int)
    argparser.add_argument('--nn_name', default='test', type=str)
    argparser.add_argument('--dilations', default=2, type=int)
    argparser.add_argument('--dropout', default=0.0, type=float)
    argparser.add_argument('--filter_size', default=9, type=int)#9
    argparser.add_argument('--architecture', default="", type=str)
    argparser.add_argument('--batchnorm_axis', default=1, type=int)
    argparser.add_argument('--batchnorm_renorm', default=0, type=int)
    argparser.add_argument('--add_drift_filter', default=1, type=int)
    argparser.add_argument('--round_decimal', default=3, type=int)
    argparser.add_argument('--subtract_output', default=1, type=int)
    argparser.add_argument('--gauss_lay', default=[], type=int, nargs='+')
    argparser.add_argument('--gauss_std', default=0.1, type=float)

    argparser.add_argument('--learning_rate', default=0.00001, type=float)
    argparser.add_argument('--opt', default='adam', type=str)
    argparser.add_argument('--pretrain_rounds', default=0, type=int)
    argparser.add_argument('--z_shortcut', default=[5], type=int, nargs='+')

    argparser.add_argument('--nn', default='E', type=str)#choices 'E' 'AE' 'VAE'
    argparser.add_argument('--pretrain', default=4000, type=int)
    argparser.add_argument('--learning_rate_pre', default=0.0005, type=float)
    argparser.add_argument('--fix_lay', default=[0,1,2,3], type=int, nargs='+')
    ##
    argparser.add_argument('--cmh_file', default='', type=str)
    argparser.add_argument('--haplotype_file', default='', type=str)
    argparser.add_argument('--selection_file', default='', type=str)



    arguments=argparser.parse_args()
    return arguments

def load_dna_cnn(args,pretrain):
    if args.non_linearity=='relu':
        non_linearity =tf.keras.activations.relu
    NUM_LAYERS = args.num_layers
    dilations = args.dilations
    filter_size = args.filter_size
    architecture = args.architecture
    dropout=args.dropout
    batchnorm_axis=args.batchnorm_axis
    batchnorm_renorm=args.batchnorm_renorm
    NUM_INIT_COLOR_CHANELS = args.num_init_color_chanels
    animal = args.animal
    MAX_GENE_LENGTH = fix_variables[animal]["max_gene_len"]
    if pretrain:
        le=args.learning_rate_pre
        pretrain=True
    else:
        le=args.learning_rate
        pretrain=False

    dna_cnn= DNA_AE(activation=non_linearity,
                     max_gene_length=MAX_GENE_LENGTH,
                     num_init_color_chanels=NUM_INIT_COLOR_CHANELS,
                     num_layers=NUM_LAYERS,
                     dilations=dilations,
                     filter_size=filter_size,
                     architecture=architecture,
                     batchnorm_axis=batchnorm_axis,
                     batchnorm_renorm=batchnorm_renorm,
                     dropout=dropout,
                     add_drift_filter=args.add_drift_filter,
                     subtract_output=args.subtract_output,
                     gauss_lay=args.gauss_lay,
                     gauss_std=args.gauss_std,
                     learning_rate=le,
                     opt=args.opt,
                    pretrain=pretrain,nn=args.nn,fix_lay=args.fix_lay,
                    z_shortcut=args.z_shortcut
                     )
    return dna_cnn

def main():


    args = parse_args()
    animal = args.animal
    Ne = fix_variables[animal]['Ne']
    Ncensus=fix_variables[animal]['Ncensus']
    Nsampling=fix_variables[animal]['Nsampling']
    generation=fix_variables[animal]['generation']
    BATCHSIZE=args.batch_size
    NUM_INIT_COLOR_CHANELS=args.num_init_color_chanels
    train_sessions=args.train_sessions
    repeat_iterations=args.repeat_iterations
    print_every=args.print_every
    work_from=args.work_from
    MAX_GENE_LENGTH = fix_variables[animal]["max_gene_len"]
    population_pairs=fix_variables[animal]["population_pairs"]
    model_path="models/" +args.nn_name
    if not os.path.exists( model_path):
        os.makedirs(model_path)
    output_1=model_path+"/output.json"
    out_1=open(output_1,'w')
    json.dump(vars(args),out_1)
    out_1.close()
    print('save params')
    print(args.nn_name)
    train_log_dir = model_path+ '/logs/'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir+'train')
    if 'AE' in args.nn:
        dataset_train = Dataset_loader_pretrain(gene_dictionary=fix_variables[work_from][animal]["train_data"],
                                             population_pairs=population_pairs,
                                             max_gene_len=MAX_GENE_LENGTH,
                                             num_init_color_channels=NUM_INIT_COLOR_CHANELS,
                                             num_gene_batches=args.num_gene_batches,
                                             num_samples=repeat_iterations, Ncensus=Ncensus, Ne=Ne, Nsampling=Nsampling,
                                             generation=generation, r_path=args.r_path, decimals=args.round_decimal)
        gene_loader_train_tmp = dataset_train.batch(BATCHSIZE).prefetch(tf.data.experimental.AUTOTUNE)
        dna_cnn =load_dna_cnn(args,pretrain=True)

        ckpt_last = tf.train.Checkpoint(step=tf.Variable(1), optimizer=dna_cnn.optimizer, net=dna_cnn)
        manager_last = tf.train.CheckpointManager(ckpt_last, model_path + '/tf_ckpts_pretrain', max_to_keep=3)
        warning_counter=0
        warning_max=25
        curr_loss_train = float('inf')
        for i in range(args.pretrain):
            ckpt_last.step.assign_add(1)
            if i % print_every == 0:
                metrics,warning = dna_cnn.pre_train(gene_loader_train_tmp, i, train_summary_writer, to_print=True)
                log_pretrain(metrics,step=i)
                if metrics[0] < curr_loss_train:
                    curr_loss_train = metrics[0]
                    save_path = manager_last.save()
                    print("Saved checkpoint TRAIN for step {}: {}".format(int(ckpt_last.step), save_path))
                if warning:
                    warning_counter += 1
                    print('WARNING zero mean abs grad:', warning_counter)
                    if warning_counter > warning_max:
                        break
                else:
                    warning_counter = 0

            else:
                dna_cnn.pre_train(gene_loader_train_tmp, i, train_summary_writer, to_print=False)
        print('pretraining end !')
    #load encoder net:
    dna_cnn = load_dna_cnn(args, pretrain=False)
    if 'AE' in args.nn:
        ckpt_tmp = tf.train.Checkpoint(step=tf.Variable(1), optimizer=dna_cnn.optimizer, net=dna_cnn)
        manager_tmp = tf.train.CheckpointManager(ckpt_tmp, model_path + '/tf_ckpts_pretrain', max_to_keep=3)
        ckpt_tmp.restore(manager_tmp.latest_checkpoint)


    if args.add_drift_filter:
        dataset_train=Dataset_loader_debug(gene_dictionary=fix_variables[work_from][animal]["train_data"],
                                    population_pairs=population_pairs,
                                    max_gene_len=MAX_GENE_LENGTH,
                                    num_init_color_channels=NUM_INIT_COLOR_CHANELS, num_gene_batches=args.num_gene_batches,
                                    num_samples=repeat_iterations,Ncensus=Ncensus,Ne=Ne,Nsampling=Nsampling,
                                    generation=generation,decimals=args.round_decimal)

    else:
        dataset_train = ArtificialDataset_debug(gene_dictionary=fix_variables[work_from][animal]["train_data"],
                                    population_pairs=population_pairs,
                                    max_gene_len=MAX_GENE_LENGTH,
                                    num_init_color_channels=NUM_INIT_COLOR_CHANELS, num_gene_batches=args.num_gene_batches,
                                    num_samples=repeat_iterations,decimals=args.round_decimal)

    gene_loader_train = dataset_train.batch(BATCHSIZE).prefetch(tf.data.experimental.AUTOTUNE)


    curr_loss_train=float('inf')
    ckpt_train = tf.train.Checkpoint(step=tf.Variable(1), optimizer=dna_cnn.optimizer, net=dna_cnn)
    manager_train = tf.train.CheckpointManager(ckpt_train, model_path + '/tf_ckpts_train', max_to_keep=3)
    ckpt_last = tf.train.Checkpoint(step=tf.Variable(1), optimizer=dna_cnn.optimizer, net=dna_cnn)
    manager_last = tf.train.CheckpointManager(ckpt_last, model_path + '/tf_ckpts_last', max_to_keep=2)


    warning_counter = 0
    warning_max = 25
    pretrain_rounds = args.pretrain_rounds
    for i in range(train_sessions+pretrain_rounds):
        ckpt_train.step.assign_add(1)
        ckpt_last.step.assign_add(1)
        
        if i % print_every == 0:

            to_print=True
            [cls_loss, acc_mean,fake_loss,fake_acc ,avg_prec,avg_prec_max,auc,auc_max,distinguish_acc],\
            warning = dna_cnn.train_dataset_debug(gene_loader_train, i, train_summary_writer,to_print=to_print,pretrain_rounds=pretrain_rounds)

            
            print(i, "train: cls_loss: {:5.4f}, accuracy: {:3.2f},  fake_loss: {:5.4f}, fake_accuracy: {:3.2f}" .format(cls_loss,  acc_mean,fake_loss,fake_acc ),flush=True)
            manager_last.save()
            if cls_loss < curr_loss_train:
                curr_loss_train=cls_loss
                save_path = manager_train.save()
                print("Saved checkpoint TRAIN for step {}: {}".format(int(ckpt_train.step), save_path))
            if warning:
                warning_counter+=1
                print('WARNING zero mean abs grad:',warning_counter)
                if warning_counter>warning_max:
                    break
            else:
                warning_counter=0
            
        else:
            dna_cnn.train_dataset_debug(gene_loader_train, i, train_summary_writer, to_print=False,pretrain_rounds=pretrain_rounds)
    return args.nn_name,work_from,animal,args


model_name,work_from,animal,args=main()
title=execute_cam(model_name,work_from,args.add_drift_filter)
if args.cmh_file!='' and args.polymorphism_file !='' and args.selection_file!='':
    cmh_file = args.cmh_file
    polymorphism_file = args.haplotype_file
    selection_file = args.selection_file
    print_result([model_name],animal,work_from,cmh_file,polymorphism_file,selection_file,title)
