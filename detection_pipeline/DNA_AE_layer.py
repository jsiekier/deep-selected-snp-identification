import tensorflow as tf
import numpy as np
from sklearn import metrics
import numpy

import ray
import psutil

num_cpus = psutil.cpu_count(logical=False)
print('-------> NUM cpus:',num_cpus)
ray.init(num_cpus=num_cpus)


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def flat_binary(input_tensor):
    input_tensor = np.asarray(input_tensor).flatten()
    input_tensor[input_tensor >= 0.5] = 1
    input_tensor[input_tensor < 0.5] = 0
    return input_tensor



def create_encoder_layers(filter_size, strides, dilations, num_filter, old_num_filter, use_shortcut, old_h, batch_ax,
                          batch_ren, var_name,trainable=True):
    conv = tf.keras.layers.Conv1D(num_filter, filter_size, dilation_rate=dilations, strides=strides, padding="same",
                                  data_format='channels_last', use_bias=False, activation=None, name=var_name,trainable=trainable)
    # dropout=tf.keras.layers.Dropout(rate=)
    batch_norm = tf.keras.layers.BatchNormalization(axis=batch_ax, renorm=batch_ren,trainable=trainable)
    if use_shortcut:
        s_pool = tf.keras.layers.AveragePooling1D(pool_size=filter_size, strides=strides, padding="same",
                                                  data_format="channels_last",trainable=trainable)
        s_dense = tf.keras.layers.Dense(1, activation=None, use_bias=True, input_shape=(-1, old_h, old_num_filter),trainable=trainable)

        return s_pool, s_dense, conv, batch_norm
    else:
        return None, None, conv, batch_norm


def create_decoder_layers(filter_size, strides, dilations, num_filter, old_num_filter, use_shortcut, old_h, batch_ax,
                          batch_ren):
    reshape_1 = tf.keras.layers.Reshape((old_h, 1, old_num_filter))  # , input_shape=(-1,old_h,old_num_filter))
    reshape_2 = tf.keras.layers.Reshape(
        (old_h * strides, old_num_filter))  # , input_shape=(-1, old_h*strides,1, old_num_filter))
    conv = tf.keras.layers.Conv1D(num_filter, filter_size, dilation_rate=dilations, strides=1, padding="same",
                                  data_format='channels_last', use_bias=False, activation=None)

    batch_norm = tf.keras.layers.BatchNormalization(axis=batch_ax, renorm=batch_ren)
    if use_shortcut:
        s_dense = tf.keras.layers.Dense(1, activation=None, use_bias=True,
                                        input_shape=(-1, old_h * strides, old_num_filter))
        return s_dense, conv, batch_norm, reshape_1, reshape_2
    else:
        return None, conv, batch_norm, reshape_1, reshape_2



def calc_loss_acc(out_loss, lab):
    output, loss = out_loss
    loss = float(loss)
    cls_output = flat_binary(output)
    lab = flat_binary(lab)
    acc = metrics.accuracy_score(lab, cls_output)
    return loss, acc


def create_guided_backprob(input, dtype='float32'):
    gradients, grad_input=input
    gate_f = tf.cast(grad_input > 0, dtype)
    gate_r = tf.cast(gradients > 0, dtype)
    gradients = tf.cast(gradients, dtype)
    return gate_f * gate_r * gradients




# @tf.function
@ray.remote(num_cpus=1)
def tmp_rank(preds):
    preds_len = len(preds)
    sorted_preds = sorted(enumerate(preds), key=lambda x: x[1])
    # sort preds acording to initial order
    preds = [(x[0] / preds_len) for x in sorted(enumerate(sorted_preds), key=lambda y: y[1])]

    return preds


def parmap(f, list):
    return [f.remote(x) for x in list]


def calc_ranking(methos_predictions, cls_output):
    methos_predictions = numpy.asarray(methos_predictions).tolist()
    result_ids = parmap(tmp_rank, methos_predictions)
    results = ray.get(result_ids)
    extreme_preds = np.asarray(results)
    return np.mean(extreme_preds, axis=0),np.mean(extreme_preds*cls_output, axis=0)
@tf.function
def sub_mean(x):
    sub=x[0] - x[1]
    mean_res=tf.keras.backend.mean(sub, axis=1, keepdims=False)
    return (sub,mean_res)
class DNA_AE(tf.keras.Model):

    def __init__(self, activation, max_gene_length, num_init_color_chanels, num_layers, dilations,
                 filter_size, architecture, batchnorm_axis, batchnorm_renorm, dropout, add_drift_filter,
                 subtract_output, learning_rate, opt
                 , gauss_lay=[],
                 gauss_std=0, use_noise=True,pretrain=False,nn='E',fix_lay=[],z_shortcut=[0,1,2,3,4,5,6,7,8]):
        super(DNA_AE, self).__init__()

        self.gauss_std = gauss_std
        self.max_gene_length = max_gene_length
        self.architecture = architecture
        self.num_init_color_chanels = num_init_color_chanels
        self.dropout = dropout
        self.dilations = dilations
        # self.out_dim=out_dim
        self.num_layers = num_layers
        if batchnorm_axis == -1:
            self.batchnorm_axis = 1
        if batchnorm_axis == 1:
            self.batchnorm_axis = -1
        else:
            self.batchnorm_axis = 0
        self.batchnorm_renorm = batchnorm_renorm
        self.add_drift_filter = add_drift_filter
        self.subtract_output = subtract_output
        self.gauss_lay = gauss_lay
        self.use_noise = use_noise
        self.pretrain=pretrain
        self.nn=nn
        self.fix_lay=fix_lay
        self.z_shortcut=z_shortcut
        self.make_init(activation, filter_size, batchnorm_axis, learning_rate, opt)

    def make_init(self, activation, filter_size, batchnorm_axis, learning_rate, opt):

        self.activation = activation  # guidedRelu#tf.keras.activations.relu#
        self.concadenate = tf.keras.layers.Concatenate(axis=-1)

        self.encoder = self.make_encoder(filter_size, batchnorm_axis)
        if self.nn=='E' or not self.pretrain:
            self.loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=False)  # tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0, #reduction=losses_utils.ReductionV2.AUTO,name='categorical_crossentropy')

            self.aling_discriminator = self.make_discriminator('align')
            self.fake_discriminator = self.make_discriminator('fake')
            # self.last_activation=tf.keras.activations.sigmoid
            self.l1_loss = tf.keras.losses.mean_absolute_error

        else:
            self.loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0,
                                                              name='categorical_crossentropy',
                                                              reduction=tf.keras.losses.Reduction.NONE)
            self.decoder = self.make_decoder(filter_size, batchnorm_axis)
            if self.nn == 'VAE':
                self.z1_mean = tf.keras.layers.Dense(self.z_dim_real, activation=None, use_bias=False,input_shape=(-1, self.z_dim_real),name='mean')
                self.z1_log_var = tf.keras.layers.Dense(self.z_dim_real, activation=None, use_bias=False,input_shape=(-1, self.z_dim_real),name='var')

        if opt == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            self.optimizer_fake = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            self.optimizer_fake2 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif opt == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            self.optimizer_fake = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            self.optimizer_fake2 = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif opt == 'rms':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            self.optimizer_fake = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            self.optimizer_fake2 = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        self.initialize_graph = False

    def make_discriminator(self, var_name):
        # print('Create',var_name)
        if self.subtract_output:
            inputs = tf.keras.Input(shape=(self.z_dim_real))
        else:
            inputs = tf.keras.Input(shape=(self.z_dim_real * 2))
        cls_output_dense = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, use_bias=False,
                                                 name=var_name)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=[cls_output_dense])
        return model

    def enc_layer(self, net, encoder_batch_input, s_pool, s_dense, conv, batch_norm, activation=None, layer_num=0):

        convolution = conv(net)
        if s_dense != None:
            shortcut = s_pool(net)
            #Reduce shortcut:
            shortcut = s_dense(shortcut)
            convolution = self.concadenate([convolution, shortcut])
        encoder_batch_input.append(convolution)
        if self.batchnorm_axis:
            convolution = batch_norm(convolution)
        if self.dropout:
            dropout = tf.keras.layers.Dropout(rate=self.dropout)
            convolution = dropout(convolution)

        convolution = self.activation(convolution)
        return convolution, encoder_batch_input



    def make_encoder(self, filter_size, batchnorm_axis):  # ,net,trainable
        # print('Create Encoder')
        inputs = tf.keras.Input(shape=(self.max_gene_length, self.num_init_color_chanels))
        net = inputs
        encoder_batch_input = []

        # ENCODER:
        strides = [2]  # ,1
        old_num_filters = 1
        old_h = self.max_gene_length
        net_layers = []
        self.z_dim_real=0
        self.mlp_start_end_list=[]
        for layer in range(self.num_layers):

            if layer in self.gauss_lay and self.gauss_std:
                net = tf.keras.layers.GaussianNoise(stddev=self.gauss_std)(net)
            num_filter = 32 * (2 ** layer)
            if num_filter > 256:
                num_filter = 256
            for stride in strides:
                if stride % 2 == 0:
                    dilat = 1
                else:
                    dilat = self.dilations
                if layer in self.fix_lay and not self.pretrain and 'A' in self.nn:
                    trainable=False
                else:
                    trainable=True
                s_pool, s_dense, conv, batch_norm = create_encoder_layers(filter_size, stride, dilat,
                                                                          num_filter, old_num_filters, True, old_h,
                                                                          batchnorm_axis, self.batchnorm_renorm,
                                                                          var_name='conv_' + str(layer),trainable=trainable)
                net, encoder_batch_input = self.enc_layer(net, encoder_batch_input, s_pool, s_dense, conv, batch_norm,
                                                          layer_num=layer)
                old_num_filters = num_filter + 1
                old_h = int(old_h / stride)
                if layer in self.z_shortcut:
                    net_layers.append(net)
                    self.z_dim_real += net.shape[-1]
                    self.mlp_start_end_list.append((self.z_dim_real-net.shape[-1],self.z_dim_real))
        self.last_filter_num = old_h#net.shape[-1]  # old_num_filters


        model = tf.keras.Model(inputs=inputs, outputs=net_layers)
        return model
    def make_decoder(self,filter_size,batchnorm_axis):
        print('Create Decoder')
        inputs = tf.keras.Input(shape=(self.z_dim_real))

        net=inputs
        net=tf.keras.layers.Dense(self.last_filter_num, activation=None, use_bias=False,input_shape=(-1,self.z_dim_real))(net)
        net= tf.reshape(net, (-1, self.last_filter_num, 1))

        add_shortcut = True
        old_num_filters = 1
        strides = [2]
        old_h=self.last_filter_num
        for layer in range(self.num_layers + 1):
            num_filter = 32 * (2 ** (self.num_layers - layer-1))
            if num_filter > 256:
                num_filter = 256
            if layer == self.num_layers:
                num_filter = self.num_init_color_chanels
                add_shortcut = False
                strides = [1]
            for stride in strides:
                if stride%2==0 or layer==self.num_layers:
                    dilat=1
                else:
                    dilat=self.dilations
                s_dense, conv, batch_norm,reshape_1,reshape_2 = create_decoder_layers(filter_size, stride,dilat,
                                                                                      num_filter,old_num_filters,
                                                                                      add_shortcut,old_h,
                                                                                      batchnorm_axis,
                                                                                      self.batchnorm_renorm)
                old_num_filters=num_filter+1
                old_h = int(old_h *stride)

                net = reshape_1(net)
                net = tf.keras.backend.resize_images(net, height_factor=stride, width_factor=1,
                                                     data_format="channels_last",
                                                     interpolation="nearest")
                net = reshape_2(net)
                convolution = conv(net)
                if s_dense != None:

                    shortcut = s_dense(net)
                    convolution = self.concadenate([convolution, shortcut])
                if self.batchnorm_axis:
                    convolution = batch_norm(convolution)

                net = self.activation(convolution)
        net=tf.keras.activations.softmax(net)
        model = tf.keras.Model(inputs=inputs, outputs=[net])
        return model

    @tf.function
    def execute_encoder(self, input1, input2, training, mean_axis=1):
        z_1 = self.encoder(input1, training=training)
        z_2 = self.encoder(input2, training=training)

        if self.subtract_output:
            if len(self.z_shortcut)==1:
                cls_input=tf.keras.backend.mean(z_1-z_2, axis=mean_axis, keepdims=False)
            else:
                cls_input=list(map(lambda x:tf.keras.backend.mean(x[0]-x[1], axis=mean_axis, keepdims=False)
                                    ,zip(z_1,z_2)))

                cls_input=tf.concat(cls_input,axis=1)        
        else:
            z_1 = tf.keras.backend.mean(z_1, axis=mean_axis, keepdims=False)
            z_2 = tf.keras.backend.mean(z_2, axis=mean_axis, keepdims=False)
            cls_input = tf.concat([z_1, z_2], axis=-1)
        return z_1, z_2, cls_input

    @tf.function
    def execute_network(self, input1, input2, training):
        # ENCODER execution 1:
        z_1, z_2, cls_input = self.execute_encoder(input1, input2, training=training)
        # CLASSIFIER:

        cls_output = self.aling_discriminator(cls_input, training=training)
        # cls_output=self.last_activation(cls_output)

        return z_1, z_2, cls_output



    def call(self, input_):
        super(DNA_AE, self).compile()
        input1, input2 = input_
        z_1, z_2, cls_input = self.execute_encoder(input1, input2, training=False)
        self.fake_discriminator(cls_input, training=False)
        self.align_discriminator(cls_input, training=False)

    @tf.function(experimental_relax_shapes=True)
    def init_graph(self, real, other, labels):
        with tf.GradientTape() as tape:
            _, _, fake_input = self.execute_encoder(real, other, training=True)
            fake_output = self.aling_discriminator(fake_input, training=True)
            fake_loss1 = self.loss(labels, fake_output)
        train_vars = self.encoder.trainable_variables + self.aling_discriminator.trainable_variables
        grads = tape.gradient(fake_loss1, train_vars)

        grads = [grads[i] * 0 for i in range(len(grads))]
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.optimizer_fake.apply_gradients(zip(grads, train_vars))
        with tf.GradientTape() as tape:
            _, _, fake_input = self.execute_encoder(real, other, training=True)
            fake_output2 = self.fake_discriminator(fake_input, training=True)
            fake_loss2 = self.loss(labels, fake_output2)
        train_vars = self.encoder.trainable_variables + self.fake_discriminator.trainable_variables
        grads = tape.gradient(fake_loss2, train_vars)
        grads = [grads[i] * 0 for i in range(len(grads))]
        self.optimizer_fake2.apply_gradients(zip(grads, train_vars))
        self.initialize_graph = True

    @tf.function(experimental_relax_shapes=True)
    def train_step_new(self, real, other, labels, discriminator, train_vars, opt, lambdas_org):
        with tf.GradientTape() as tape:
            _, _, fake_input = self.execute_encoder(real, other, training=True)
            fake_output = discriminator(fake_input, training=True)
            fake_loss = self.loss(labels, fake_output)
        grads = tape.gradient(fake_loss, train_vars)

        grads = [grads[i] * lambdas_org[i] for i in range(len(grads))]
        opt.apply_gradients(zip(grads, train_vars))
        return fake_output, fake_loss

    def new_train_function(self, tensor_input):
        res_dict = dict()
        if self.add_drift_filter:
            real, other, fake_real, fake_other, adaptation_labels, fake_inst, real_inst, fake_labels, five_labs, step, to_print = tensor_input
            if not self.initialize_graph:
                self.init_graph(real, other, adaptation_labels)

            pretrain_step = 0 if step < self.pretrain_rounds else step - self.pretrain_rounds

            train_vars = self.encoder.trainable_variables + self.aling_discriminator.trainable_variables
            if self.add_drift_filter:
                lambdas = tf.convert_to_tensor(np.asarray([1.0] * len(train_vars)), dtype=tf.dtypes.float32)
            else:
                lambdas = tf.convert_to_tensor(np.asarray([1.0] * len(train_vars)), dtype=tf.dtypes.float32)
            output, loss = self.train_step_new(fake_inst, real_inst, fake_labels, self.aling_discriminator,
                                               train_vars,
                                               self.optimizer, lambdas)
            res_dict['align_order'] = (output, loss)



        else:
            real, other, adaptation_labels = tensor_input
            train_vars = self.encoder.trainable_variables + self.aling_discriminator.trainable_variables
            lambdas = tf.convert_to_tensor(np.asarray([1.0] * len(train_vars)), dtype=tf.dtypes.float32)
            output, loss = self.train_step_new(real, other, adaptation_labels, self.aling_discriminator,
                                               train_vars,
                                               self.optimizer, lambdas)
            res_dict['align_order'] = (output, loss)
        return res_dict



    @tf.function
    def cam_execution(self, input1_, input2_, training):
        input1_ = tf.convert_to_tensor(input1_)
        input2_ = tf.convert_to_tensor(input2_)
        c = tf.concat([input1_, input2_], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(c)
            # concat = guidedRelu(c)
            input1 = c[:, :self.max_gene_length]
            input2 = c[:, self.max_gene_length:]
            z1 = self.encoder(input1, training=training)
            z2 = self.encoder(input2, training=training)
            # CLASSIFIER:
            #subtract_vec = z1 - z2
            subtract_vec, cls_input=[],[]
            if len(self.z_shortcut)==1:
                sub = z1 - z2
                mean_res = tf.keras.backend.mean(sub, axis=1, keepdims=False)
                subtract_vec.append(sub)
                cls_input.append(mean_res)

            else:
                for x in zip(z1,z2):
                    sub = x[0] - x[1]
                    mean_res = tf.keras.backend.mean(sub, axis=1, keepdims=False)
                    subtract_vec.append(sub)
                    cls_input.append(mean_res)            #sub_mean_list=list(map(sub_mean,zip(z1,z2)))
            #subtract_vec, cls_input=zip(*sub_mean_list)
            #cls_input = z_1 - z_2
            #cls_input = tf.keras.backend.mean(cls_input, axis=mean_axis, keepdims=False)
            cls_input=tf.concat(cls_input,axis=1)
            #subtract_vec=list(subtract_vec)
            #cls_input = tf.keras.backend.mean(subtract_vec, axis=1, keepdims=False)
            cls_output = self.aling_discriminator(cls_input, training=training)
        all_grads = tape.gradient(cls_output, subtract_vec+[ c])
        guided_backprob = create_guided_backprob((all_grads[-1], c))
        guided_backprob2 = list(map(create_guided_backprob,zip(all_grads, subtract_vec) ))

        return cls_output, subtract_vec, guided_backprob, guided_backprob2

    def create_heat_map_subtract_batch_single_lay(self,input):
        vec ,w_s_tmp=input
        all_heats = tf.reduce_mean(vec * w_s_tmp, axis=2)
        heat_transform = tf.reshape(all_heats, [all_heats.shape[0], all_heats.shape[-1], 1, 1])
        heat_resized = tf.image.resize(heat_transform, size=[self.max_gene_length, 1],
                                       method=tf.image.ResizeMethod.BILINEAR)
        heat_resized = tf.reshape(heat_resized, [all_heats.shape[0], self.max_gene_length])
        return heat_resized
    def create_heat_map_subtract_batch(self, vec, gap_w):
        w = tf.transpose(gap_w, (1, 0))
        w_s_tmp = tf.expand_dims(w, axis=0)
        #TODO aufteilung von w entsprechend der maps'
        w_s_tmps=[w_s_tmp[:,:,start:end] for (start,end) in self.mlp_start_end_list]
        heat_resized_all=list(map(self.create_heat_map_subtract_batch_single_lay,zip(vec,w_s_tmps)))
        heat_resized=tf.reduce_mean(tf.convert_to_tensor(heat_resized_all),axis=0,keepdims=False)
        return heat_resized,heat_resized_all

    def create_org_grad_cam_batch_single_lay(self,input):
        guided_backprob2, subtract_vec=input

        weights = tf.reduce_mean(guided_backprob2, axis=1,keepdims=True)
        #w_tmp = tf.expand_dims(weights, axis=1)
        # divide by number of color chanels! to get equal weight for each layer
        all_heats = tf.reduce_mean(weights * subtract_vec, axis=2)
        heat_transform = tf.reshape(all_heats, [all_heats.shape[0], all_heats.shape[-1], 1, 1])
        heat_resized = tf.image.resize(heat_transform, size=[self.max_gene_length, 1],
                                       method=tf.image.ResizeMethod.BILINEAR)
        all_cams = tf.reshape(heat_resized, [all_heats.shape[0], self.max_gene_length])
        return all_cams
    def create_org_grad_cam_batch(self, guided_backprob2, subtract_vec):
        # with tf.device(device):
        all_cams_layer=list(map(self.create_org_grad_cam_batch_single_lay,zip(guided_backprob2, subtract_vec)))
        all_cams=tf.convert_to_tensor(all_cams_layer)
        all_cams=tf.reduce_mean(all_cams,axis=0,keepdims=False)
        return all_cams,all_cams_layer

    def summarize(self, all_cams, cls_output):
        cam_mean = tf.reduce_mean(all_cams, axis=0)
        cam_ranking,cam_ranking_w = calc_ranking(all_cams, cls_output)
        #cam_median = numpy.median(numpy.asarray(all_cams), axis=0)
        cam_w = tf.reduce_mean(cls_output * all_cams, axis=0)
        return cam_mean,cam_ranking, cam_w,cam_ranking_w
    def create_ranking_list(self,all_cams_layer):
        all_cams_ranking_layer = list(map(calc_ranking, all_cams_layer))
        all_cams_ranking_layer=np.transpose(np.asarray(all_cams_ranking_layer),(1,0))
        return all_cams_ranking_layer
    def create_ranking_list_(self,all_cams_layer,guided_backprob):
        tmp=[]
        for entry in all_cams_layer:
            tmp.append(entry*guided_backprob)
        all_cams_layer=tmp
        all_cams_ranking_layer = list(map(calc_ranking, all_cams_layer))
        all_cams_ranking_layer=np.transpose(np.asarray(all_cams_ranking_layer),(1,0))
        return all_cams_ranking_layer

    def get_cam_subtract_batch(self, input1_, input2_, training=False):
        cls_output, subtract_vec, guided_backprob, guided_backprob2 = self.cam_execution(input1_, input2_, training)

        cam_org,all_cams_layer = self.create_org_grad_cam_batch(guided_backprob2, subtract_vec)


        cam_mean,  cam_ranking, org_grad_cam_w,org_cam_ranking_w = self.summarize(cam_org, cls_output)
        heat_resized_0,heat_resized_all = self.create_heat_map_subtract_batch(subtract_vec,
                                                             self.aling_discriminator.trainable_variables[0])
        guided_backprob = tf.reduce_max(guided_backprob, axis=-1)
        guided_backprop_0 = guided_backprob[:, :self.max_gene_length] + guided_backprob[:, self.max_gene_length:]
        org_grad_cam_ranking_lay = []
        result_0 = guided_backprop_0 * heat_resized_0
        result_0,  result_ranking, result_w,cam_ranking_w = self.summarize(result_0, cls_output)
        # cam_time=time.time()-start_time
        return result_0, cls_output, np.mean(heat_resized_0, axis=0), np.mean(guided_backprop_0, axis=0), \
               cam_mean, cam_ranking, result_ranking, org_grad_cam_w, result_w,org_grad_cam_ranking_lay,org_cam_ranking_w,cam_ranking_w  # ,net_time,cam_time,cam_org





    def test_dataset_debug(self, dataset, summary_writer, step):
        entropy_cls = 0.0
        accurancy_cls = 0.0
        drift_filter_acc = 0.0
        drift_filter_entro = 0.0
        num_batches = 0
        predictions_arr = []
        selections_arr = []
        if self.add_drift_filter:

            for real, other, real_fake, other_fake, lab, selection_lab,r1, r2, rl, _ in dataset:
                z_1, z_2, cls_output = self.execute_network(r1, r2, training=False)
                cls_entro = float(self.loss(rl, cls_output))
                entropy_cls += cls_entro
                lab_flat = flat_binary(rl)

                z_1_fake, z_2_fake, discr_output = self.execute_network(real_fake, other_fake, training=False)
                drift_filter_entro += (float(self.loss(lab, discr_output)))
                fake_predictions = flat_binary(discr_output)
                drift_filter_acc += metrics.accuracy_score(lab_flat, fake_predictions)

                # add AUC mean precision values  for debugging
                cls_output_flat = np.asarray(cls_output).flatten().tolist()
                tmp_arr = [cls_output_flat[i] if rl[i] else 1 - cls_output_flat[i] for i in
                           range(len(cls_output_flat))]
                predictions_arr.extend(tmp_arr)
                selections_arr.extend(flat_binary(selection_lab))
                cls_output = flat_binary(cls_output)
                accurancy_cls += metrics.accuracy_score(lab_flat, cls_output)
                num_batches += 1



        else:
            for real, other, lab, selection_lab in dataset:
                z_1, z_2, cls_output = self.execute_network(real, other, training=False)
                cls_entro = float(self.loss(lab, cls_output))
                entropy_cls += cls_entro
                # add AUC mean precision values  for debugging
                cls_output_flat = np.asarray(cls_output).flatten().tolist()
                tmp_arr = [cls_output_flat[i] if lab[i] else 1 - cls_output_flat[i] for i in
                           range(len(cls_output_flat))]
                predictions_arr.extend(tmp_arr)
                selections_arr.extend(flat_binary(selection_lab))
                cls_output = flat_binary(cls_output)
                accurancy_cls += metrics.accuracy_score(lab, cls_output)
                num_batches += 1
                # add AUC mean precision values  for debugging
        if sum(selections_arr):
            align_preds_max = [e if e > 0.5 else 1 - e for e in predictions_arr]
            average_precision = metrics.average_precision_score(selections_arr, predictions_arr)
            average_precision_max = metrics.average_precision_score(selections_arr, align_preds_max)
            fpr_p, tpr_p, _ = metrics.roc_curve(selections_arr, predictions_arr, pos_label=1)
            roc_auc_p = metrics.auc(fpr_p, tpr_p)
            fpr_max, tpr_max, _ = metrics.roc_curve(selections_arr, align_preds_max, pos_label=1)
            roc_auc_max = metrics.auc(fpr_max, tpr_max)
        else:
            average_precision, average_precision_max, roc_auc_p, roc_auc_max = 0, 0, 0, 0
        output = [entropy_cls / num_batches, accurancy_cls / num_batches, drift_filter_entro / num_batches,
                  drift_filter_acc / num_batches]
        output += [average_precision, average_precision_max, roc_auc_p, roc_auc_max]
        self.write_statistics(summary_writer, step, [], output)
        return output

    @tf.function
    def pretrain_step(self,x0,x0_mask):
        training=True
        with tf.GradientTape() as tape:
            # encode
            z_1 = self.encoder(x0, training=training)
            # make z
            if len(self.z_shortcut)==1:
                cls_input = tf.keras.backend.mean(z_1, axis=1, keepdims=False)
            else:
                cls_input = list(map(lambda x: tf.keras.backend.mean(x, axis=1, keepdims=False)
                                 , z_1))
                cls_input = tf.concat(cls_input, axis=1)            # variational
            if self.nn=='VAE':
                cls_input_mean =self.z1_mean(cls_input)
                cls_input_var =self.z1_log_var(cls_input)
                eps = tf.random.normal(shape=cls_input_mean.shape, mean=0, stddev=1)
                cls_input=eps * tf.exp(cls_input_var * .5) + cls_input_mean#eps * (1e-8 + cls_input_var) +cls_input_mean
            # decode
            x0_new=self.decoder(cls_input,training=training)
            tmp1=tf.reduce_sum(self.loss(x0, x0_new),axis=-1)
            tmp2=tf.reduce_sum(x0_mask,axis=(-1,-2))
            entro_loss = tf.reduce_mean(tmp1/tmp2)
            #TODO mask the loss!!!
            kl_loss=0
            if self.nn =='VAE':
                kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + cls_input_var - tf.square(cls_input_mean) - tf.exp(cls_input_var), axis=1))
                loss=entro_loss+kl_loss
            else:
                loss=entro_loss
        train_vars=self.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return x0_new,loss,entro_loss,kl_loss,grads





    # @tf.function
    def write_statistics(self, summary_writer, step, grads, statistics):
        warning = False
        other_vars = self.encoder.trainable_variables + self.aling_discriminator.trainable_variables
        with summary_writer.as_default():
            for g, var in zip(grads, other_vars):
                mean_grad = tf.reduce_mean(tf.abs(g))
                mean_var = tf.reduce_mean(tf.abs(var))
                tf.summary.scalar(var.name + '_mean abs grad', mean_grad, step=step)
                tf.summary.scalar(var.name + '_mean abs variable', mean_var, step=step)
                if not mean_grad:
                    warning = True

            for var in self.aling_discriminator.trainable_variables:
                mean_var = tf.reduce_mean(tf.abs(var))
                tf.summary.scalar(var.name + '_mean abs variable', mean_var, step=step)

            for name, stat in zip(['align discrimination entropy', 'align accuracy', 'fake entropy',
                                   'fake accuracy', 'average selection precision', 'average selection precision max',
                                   'selection AUC', 'selection AUC max'], statistics):
                tf.summary.scalar('statistics/' + name, stat, step=step)

        return warning


    def write_pretrain_statistics(self, summary_writer, step, grads, statistics):
        warning = False
        other_vars = self.trainable_variables
        with summary_writer.as_default():
            for g, var in zip(grads, other_vars):
                mean_grad = tf.reduce_mean(tf.abs(g))
                mean_var = tf.reduce_mean(tf.abs(var))
                tf.summary.scalar(var.name + '_mean abs grad', mean_grad, step=step)
                tf.summary.scalar(var.name + '_mean abs variable', mean_var, step=step)
                if not mean_grad:
                    warning = True
            for name, stat in zip(['loss','entropy','kld','l1_reinforce'], statistics):
                tf.summary.scalar('statistics/' + name, stat, step=step)

        return warning

    def pre_train(self, dataset, step, summary_writer=None, to_print=True):
        entro_loss=0
        all_loss=0
        kl_loss=0
        l1_loss=0
        counter=0
        # if self.add_drift_filter:
        for (x0) in dataset:
            x0_mask=np.sum(x0,keepdims=True,axis=-1)
            x0_new,loss,entropy,kl_l,grads = self.pretrain_step(x0,x0_mask)
            if to_print:
                entro_loss+=entropy
                kl_loss+=kl_l
                all_loss+=loss
                l1_loss+= calc_reinforced_l1_loss(x0, x0_new,x0_mask)
                counter += 1
        if to_print:
            output = [all_loss/counter,entro_loss / counter,kl_loss/counter, l1_loss/counter]
            warning = self.write_pretrain_statistics(summary_writer, step, grads, output)
            return output, warning

    def train_dataset_debug(self, dataset, step, summary_writer=None, to_print=True, pretrain_rounds=0):
        distinguish_acc = 0.0
        distionguish_entro = 0.0
        entropy_cls = 0.0
        accurancy_cls = 0.0
        drift_filter_acc = 0.0
        drift_filter_entro = 0.0
        counter = 0
        predictions_arr = []
        selections_arr = []
        self.pretrain_rounds = pretrain_rounds
        # if self.add_drift_filter:
        for dataset_out in dataset:
            if self.add_drift_filter:
                real, other, real_fake, other_fake, lab, selection_lab, fake_inst, real_inst, fake_labels, five_lab = dataset_out
                res_dict = self.new_train_function((real, other, real_fake, other_fake, lab, fake_inst, real_inst,
                                                    fake_labels, five_lab, step, to_print))
            else:
                real, other, lab, selection_lab = dataset_out
                res_dict = self.new_train_function(dataset_out)
            # cls_output, cls_loss, fake_output, fake_loss,grads = self.train_step([real, other,real_fake,other_fake, lab,step,to_print,fake_inst,real_inst,fake_labels])

            if to_print:
                if 'align_order' in res_dict:
                    output, loss = res_dict['align_order']
                    loss, acc = calc_loss_acc((output, loss), fake_labels)#lab
                    entropy_cls += loss
                    accurancy_cls += acc
                    # add AUC mean precision values  for debugging
                    cls_output_flat = np.asarray(output).flatten().tolist()
                    tmp_arr = [cls_output_flat[i] if fake_labels[i] else 1 - cls_output_flat[i] for i in
                               range(len(cls_output_flat))]#lab
                    predictions_arr.extend(tmp_arr)
                    selections_arr.extend(flat_binary(selection_lab))
                if 'align_order_fake' in res_dict:
                    loss, acc = calc_loss_acc(res_dict['align_order_fake'], lab)
                    drift_filter_entro += loss
                    drift_filter_acc += acc
                if 'fake_org_order' in res_dict:
                    loss, acc = calc_loss_acc(res_dict['fake_org_order'], fake_labels)
                    distionguish_entro += loss
                    distinguish_acc += acc


                counter += 1

        if to_print:
            num_batches = counter

            if sum(selections_arr):
                align_preds_max = [e if e > 0.5 else 1 - e for e in predictions_arr]
                average_precision = metrics.average_precision_score(selections_arr, predictions_arr)
                average_precision_max = metrics.average_precision_score(selections_arr, align_preds_max)
                fpr_p, tpr_p, _ = metrics.roc_curve(selections_arr, predictions_arr, pos_label=1)
                roc_auc_p = metrics.auc(fpr_p, tpr_p)
                fpr_max, tpr_max, _ = metrics.roc_curve(selections_arr, align_preds_max, pos_label=1)
                roc_auc_max = metrics.auc(fpr_max, tpr_max)
            else:
                average_precision, average_precision_max, roc_auc_p, roc_auc_max = 0, 0, 0, 0

            output = [entropy_cls / num_batches, accurancy_cls / num_batches, drift_filter_entro / num_batches,
                      drift_filter_acc / num_batches]
            output += [average_precision, average_precision_max, roc_auc_p, roc_auc_max, distinguish_acc]
            warning = self.write_statistics(summary_writer, step, [], output)
            return output, warning



def calc_reinforced_l1_loss(real, softmax_output,x0_mask):
    l1_mean=0
    gene_length,gene_with=softmax_output.shape[1],softmax_output.shape[2]
    for r,s,mask in zip(real,softmax_output,x0_mask):
        arg_max=np.argmax(s,axis=1)
        new_s=np.zeros((gene_length,gene_with))
        for counter, entry in enumerate(arg_max.tolist()):
            new_s[counter,entry]=1.0
        tmp=np.abs(new_s-r)
        tmp=tmp*mask
        l1_mean+=(np.sum(tmp)/np.sum(mask))
    return l1_mean/real.shape[0]