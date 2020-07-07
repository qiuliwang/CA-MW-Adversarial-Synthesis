from utils import save_images, vis_square,sample_label,sample_masks, sample_masks_test
from tensorflow.contrib.layers.python.layers import xavier_initializer
import cv2
from ops import *
import tensorflow as tf
import numpy as np

class CMGAN(object):

    # build model
    def __init__(self, data_ob, train_dir, eval_dir, test_dir, output_size, learn_rate, batch_size, z_dim, y_dim, log_dir
         , model_path, load = False, gf_dim=64, df_dim = 64, output_c_dim=1, L1_lambda=100):

        self.data_ob = data_ob
        self.train_dir = train_dir
        self.eval_dir = eval_dir
        self.test_dir = test_dir
        self.output_size = output_size
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.load = load
        self.log_dir = log_dir
        self.model_path = model_path
        self.channel = self.data_ob.shape[2]
        self.images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.masks = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.lungwindow = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.mediastinumwindow = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])

        self.L1_lambda = L1_lambda
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim])
        self.training_step = 50000

        print('image shape: ', self.images.get_shape().as_list())

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.output_c_dim = output_c_dim
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

    def build_model(self):
        print('Building the model:')
        self.real_A = self.masks
        print('shape of real_A: ', self.real_A.get_shape().as_list())


        self.fake_B = self.generator(self.real_A, self.y)
        print('shape of self.y: ', self.y.get_shape().as_list())
        print('shape of fake_B: ', self.fake_B.get_shape().as_list())

        self.lung_logits, self.fake_lungwindow = self.decorator_lung_window(self.fake_B, self.lungwindow, self.y)
        self.mediastinum_logits, self.fake_mediastinumwindow = self.decorator_mediastinum_window(self.fake_B, self.mediastinumwindow, self.y)
        self.real_all_masks_images = tf.concat([self.real_A, self.images], 3)
        self.fake_all_masks_images = tf.concat([self.real_A, self.fake_B], 3)

        self.D_all, self.D_all_logits = self.discriminator_all(self.real_all_masks_images, self.y, reuse = False)
        self.D_all_, self.D_all_logits_ = self.discriminator_all(self.fake_all_masks_images, self.y, reuse = True)

        self.d_all_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_all_logits, labels = tf.ones_like(self.D_all)))
        self.d_all_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_all_logits_, labels = tf.zeros_like(self.D_all_)))
        
        self.y_lobu = tf.slice(self.y, [0,0], [64,5])
        self.y_spicu = tf.slice(self.y, [0,5], [64,5])
        self.y_mali = tf.slice(self.y, [0,10], [64,5])
        print('shape of lobu label ', self.y_lobu.get_shape().as_list())
        print('shape of spicu label ', self.y_spicu.get_shape().as_list())
        print('shape of mali label ', self.y_mali.get_shape().as_list())

        # the loss of classify
        self.pre_lung, self.pre_lung_logits = self.classify_lung(self.fake_lungwindow)
        self.d_loss_classify_lung = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pre_lung, labels=self.y_spicu))

        self.pre_mediastinum, self.pre_mediastinum_logits = self.classify_mediastinum(self.fake_mediastinumwindow)
        self.d_loss_classify_mediastinum = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pre_mediastinum, labels=self.y_lobu))

        # the loss of malignancy prediction
        self.pre_lung_mali, self.pre_lung_logits_mali = self.classify_lung_mali(self.fake_lungwindow)
        self.d_loss_classify_lung_mali = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pre_lung_mali, labels=self.y_mali))

        self.pre_mediastinum_mali, self.pre_mediastinum_logits_mali = self.classify_mediastinum_mali(self.fake_mediastinumwindow)
        self.d_loss_classify_mediastinum_mali = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pre_mediastinum_mali, labels=self.y_mali))


        #the loss of generator network
        # TODO tf.reduce_sum()
        self.g_lung_loss = self.L1_lambda * tf.reduce_mean(tf.square(self.lungwindow - self.fake_lungwindow))
        
        self.g_mediastinum_loss = self.L1_lambda * tf.reduce_mean(tf.square(self.mediastinumwindow - self.fake_mediastinumwindow))
        
        self.g_all_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_all_logits_, labels=tf.ones_like(self.D_all_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.square(self.images - self.fake_B))

        self.d_all_loss = self.d_all_loss_fake + self.d_all_loss_real + self.d_loss_classify_lung + self.d_loss_classify_mediastinum + self.d_loss_classify_lung_mali + self.d_loss_classify_mediastinum_mali
        self.d_c_loss = self.d_loss_classify_lung + self.d_loss_classify_mediastinum + self.d_loss_classify_lung_mali + self.d_loss_classify_mediastinum_mali 
        self.d_loss = self.d_all_loss
        self.g_loss = self.g_lung_loss + self.g_mediastinum_loss + self.g_all_loss

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.c_vars = [var for var in t_vars if 'c_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)

    def train(self,args):
    
        opti_D = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        opti_C = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_c_loss, var_list=self.c_vars)

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            # self.g_sum = tf.summary.merge()

            if self.load:
                print('loading:')
                self.saver = tf.train.import_meta_graph('./model/model.ckpt-20201.meta')  # default to save all variable
                self.saver.restore(sess, tf.train.latest_checkpoint('./model/'))
            self.writer = tf.summary.FileWriter("./logs", sess.graph)

            summary_writer = tf.summary.FileWriter(self.log_dir, graph=sess.graph)

            step = 0
            while step <= self.training_step:
                realbatch_array, real_lungs, real_mediastinums, realmasks, real_labels = self.data_ob.getNext_batch(step,batch_size=self.batch_size)
                batch_z = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim])
                sess.run([opti_D],feed_dict={self.images: realbatch_array, self.lungwindow: real_lungs, self.mediastinumwindow: real_mediastinums, self.masks: realmasks, self.z: batch_z, self.y: real_labels})

                sess.run([opti_G],feed_dict={self.images: realbatch_array, self.lungwindow: real_lungs, self.mediastinumwindow: real_mediastinums, self.masks: realmasks, self.z: batch_z, self.y: real_labels})
                sess.run([opti_C],feed_dict={self.images: realbatch_array, self.lungwindow: real_lungs, self.mediastinumwindow: real_mediastinums, self.masks: realmasks, self.z: batch_z, self.y: real_labels})
                    

                if np.mod(step, 50) == 1 and step != 0:
                    print('Training...')
                    sample_images, lungwindow, mediastinumwindow = sess.run([self.fake_B, self.fake_lungwindow, self.fake_mediastinumwindow], feed_dict={self.images: realbatch_array, self.lungwindow: real_lungs, self.mediastinumwindow: real_mediastinums, self.masks: realmasks, self.z: batch_z, self.y: real_labels})
                    save_images(sample_images, [8, 8],
                                './{}/{:04d}_sample.png'.format(self.train_dir, step))
                    save_images(lungwindow, [8, 8],
                                './{}/{:04d}_lung.png'.format(self.train_dir, step))
                    save_images(mediastinumwindow, [8, 8],
                                './{}/{:04d}_mediastinum.png'.format(self.train_dir, step))

                    save_images(realmasks, [8, 8],
                                './{}/{:04d}_mask.png'.format(self.train_dir, step)) 
                                
                    print('save eval image')

                    real_labels = sample_label()
                    realmasks = sample_masks()

                    sample_images, lungwindow, mediastinumwindow = sess.run([self.fake_B, self.fake_lungwindow, self.fake_mediastinumwindow], feed_dict={self.masks: realmasks, self.y: real_labels})

                    save_images(sample_images, [8, 8],
                                './{}/{:04d}_sample.png'.format(self.eval_dir, step))
                    save_images(lungwindow, [8, 8],
                                './{}/{:04d}_lung.png'.format(self.eval_dir, step))
                    save_images(mediastinumwindow, [8, 8],
                                './{}/{:04d}_mediastinum.png'.format(self.eval_dir, step))  
                    save_images(realmasks, [8, 8],
                                './{}/{:04d}_mask.png'.format(self.eval_dir, step)) 
                    
                    #================
                    print('save test image')

                    real_labels = sample_label()
                    realmasks = sample_masks_test()

                    sample_images, lungwindow, mediastinumwindow = sess.run([self.fake_B, self.fake_lungwindow, self.fake_mediastinumwindow], feed_dict={self.masks: realmasks, self.y: real_labels})

                    save_images(sample_images, [8, 8],
                                './{}/{:04d}_sample.png'.format(self.test_dir, step))
                    save_images(lungwindow, [8, 8],
                                './{}/{:04d}_lung.png'.format(self.test_dir, step))
                    save_images(mediastinumwindow, [8, 8],
                                './{}/{:04d}_mediastinum.png'.format(self.test_dir, step))                   
                    save_images(realmasks, [8, 8],
                                './{}/{:04d}_mask.png'.format(self.test_dir, step))  
                    self.saver.save(sess, self.model_path,global_step=step)

                step = step + 1

            save_path = self.saver.save(sess, self.model_path)
            print ("Model saved in file: %s" % save_path)

    def test(self):

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            self.saver.restore(sess, self.model_path)
            sample_z = np.random.uniform(1, -1, size=[self.batch_size, self.z_dim])

            output = sess.run(self.fake_images, feed_dict={self.z: sample_z, self.y: sample_label()})

            save_images(output, [8, 8], './{}/test{:02d}_{:04d}.png'.format(self.train_dir, 0, 0))

            image = cv2.imread('./{}/test{:02d}_{:04d}.png'.format(self.train_dir, 0, 0), 0)

            cv2.imshow("test", image)

            cv2.waitKey(-1)

            print("Test finish!")

    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:
            print('generator U-Net:')
            print('shape of y: ', y.get_shape().as_list()) # 64 * 13
            print('shape of image: ', image.get_shape().as_list()) # 64 * 128 * 128 * 1
            y = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
            
            print('shape of image: ', image.get_shape().as_list())
           
            e1 = conv2d_UNet(image, self.gf_dim, name='g_e1_conv')
            e1 = conv_cond_concat(e1, y)

            e2 = self.g_bn_e2(conv2d_UNet(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            e2 = conv_cond_concat(e2, y)

            e3 = self.g_bn_e3(conv2d_UNet(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            e3 = conv_cond_concat(e3, y)

            e4 = self.g_bn_e4(conv2d_UNet(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            e4 = conv_cond_concat(e4, y)

            e5 = self.g_bn_e5(conv2d_UNet(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            e5 = conv_cond_concat(e5, y)
     
            e6 = self.g_bn_e6(conv2d_UNet(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            e6 = conv_cond_concat(e6, y)

            e7 = self.g_bn_e7(conv2d_UNet(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            e7 = conv_cond_concat(e7, y)

            e8 = self.g_bn_e8(conv2d_UNet(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            e8 = conv_cond_concat(e8, y)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            d1 = conv_cond_concat(d1, y)
           
            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            d2 = conv_cond_concat(d2, y)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            d3 = conv_cond_concat(d3, y)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            d4 = conv_cond_concat(d4, y)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            d5 = conv_cond_concat(d5, y)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            d6 = conv_cond_concat(d6, y)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            d7 = conv_cond_concat(d7, y)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            print('shape of d8: ', self.d8.get_shape().as_list())
            return tf.nn.tanh(self.d8)

    def decorator_lung_window(self, images, masks, y = None):
        with tf.variable_scope('decorator_lungwindow') as scope:
            print('Decorator for lung window')
            y = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])

            images = conv_cond_concat(images, y)
            h0 = lrelu(conv2d_decorator(images, 32, name='g_deco_lung_h0_conv'))
            h0 = conv_cond_concat(h0, y)
            h1 = lrelu(conv2d_decorator(h0, 64, name='g_deco_lung_h1_conv'))
            h1 = conv_cond_concat(h1, y)
            h2 = lrelu(conv2d_decorator(h1, 128, name = 'g_deco_lung_h2_conv'))
            h2 = conv_cond_concat(h2, y)
            h3 = lrelu(conv2d_decorator(h2, 64, name = 'g_deco_lung_h3_conv'))
            h3 = conv_cond_concat(h3, y)
            h4 = lrelu(conv2d_decorator(h3, 32, name='g_deco_lung_h4_conv'))
            h4 = conv_cond_concat(h4, y)
            h5 = lrelu(conv2d_decorator(h4, self.output_c_dim, name='g_deco_lung_h5_conv'))
            print('shape of h5: ', h5.get_shape().as_list())
            return tf.nn.relu(h5), h5

    def decorator_mediastinum_window(self, images, masks, y = None):
        with tf.variable_scope('decorator_mediastinum') as scope:
            print('Decorator for mediastinum window')
            y = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])

            images = conv_cond_concat(images, y)
            h0 = lrelu(conv2d_decorator(images, 32, name='g_deco_mediastinum_h0_conv'))
            h0 = conv_cond_concat(h0, y)
            h1 = lrelu(conv2d_decorator(h0, 64, name='g_deco_mediastinum_h1_conv'))
            h1 = conv_cond_concat(h1, y)
            h2 = lrelu(conv2d_decorator(h1, 128, name = 'g_deco_mediastinum_h2_conv'))            
            h2 = conv_cond_concat(h2, y)
            h3 = lrelu(conv2d_decorator(h2, 64, name = 'g_deco_mediastinum_h3_conv'))            
            h3 = conv_cond_concat(h3, y)
            h4 = lrelu(conv2d_decorator(h3, 32, name='g_deco_mediastinum_h4_conv'))            
            h4 = conv_cond_concat(h4, y)
            h5 = lrelu(conv2d_decorator(h4, self.output_c_dim, name='g_deco_mediastinum_h5_conv'))
            print('shape of h5: ', h5.get_shape().as_list())
            return tf.nn.relu(h5), h5

    def decorator(self, images, masks, y = None):
        with tf.variable_scope('decorator_allattributes') as scope:
            print('Decorator for all attributes')
            y = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])

            images = conv_cond_concat(images, y)

            h0 = lrelu(conv2d_decorator(images, 32, name='g_deco_h0_conv'))
            h0 = conv_cond_concat(h0, y)
            h1 = lrelu(conv2d_decorator(h0, 64, name='g_deco_h1_conv'))
            h1 = conv_cond_concat(h1, y)
            h2 = lrelu(conv2d_decorator(h1, 128, name = 'g_deco_h2_conv'))
            h2 = conv_cond_concat(h2, y)
            h3 = lrelu(conv2d_decorator(h2, 64, name = 'g_deco_h3_conv'))
            h3 = conv_cond_concat(h3, y)
            h4 = lrelu(conv2d_decorator(h3, 32, name='g_deco_h4_conv'))
            h4 = conv_cond_concat(h4, y)
            h5 = lrelu(conv2d_decorator(h4, self.output_c_dim, name='g_deco_h5_conv'))
            print('shape of h5: ', h5.get_shape().as_list())
            return tf.nn.tanh(h5), h5

    def discriminator_all(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            print('discriminator:')
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            y = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])

            image = conv_cond_concat(image, y)
            h0 = lrelu(conv2d_UNet(image, self.df_dim, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, y)
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d_UNet(h0, self.df_dim*2, name='d_h1_conv')))
            h1 = conv_cond_concat(h1, y)
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d_UNet(h1, self.df_dim*4, name='d_h2_conv')))
            h2 = conv_cond_concat(h2, y)
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d_UNet(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            h3 = conv_cond_concat(h3, y)
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            print('shape of h4: ', h4.get_shape().as_list())

            return tf.nn.sigmoid(h4), h4

    def classify_lung(self, image, y=None, reuse=False):
        with tf.variable_scope("classify_lung") as scope:
            print('classify_lung:')
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            layer = tf.layers.conv2d(image,64,[3,3],padding="same",activation=tf.nn.relu,name='c_classify_1')
            layer = tf.layers.max_pooling2d(layer,pool_size=[2,2],strides=2,name='c_classify_2')
            layer = tf.layers.conv2d(layer,128,[3,3],padding="same",activation=tf.nn.relu,name='c_classify_3')
            layer = tf.layers.max_pooling2d(layer,pool_size=[2,2],strides=2,name='c_classify_4')

            layer = tf.layers.conv2d(layer,256,[3,3],padding="same",activation=tf.nn.relu,name='c_classify_5')
            layer = tf.layers.max_pooling2d(layer,pool_size=[2,2],strides=2,name='c_classify_6')
            layer = tf.reshape(layer, [-1, 16 * 16 * 256])

            layer = tf.layers.dense(layer,100,activation=tf.nn.relu,name='c_classify_7')
            layer = tf.layers.dropout(layer,0.5,name='c_classify_8')
            layer = tf.layers.dense(layer,5,activation=tf.nn.relu,name='c_classify_9')
            logits = tf.nn.softmax(layer)
            return layer, logits

    def classify_mediastinum(self, image, y=None, reuse=False):
        with tf.variable_scope("classify_mediastinum") as scope:
            print('classify_mediastinum:')
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            layer = tf.layers.conv2d(image,64,[3,3],padding="same",activation=tf.nn.relu,name='c_mediastinum_1')
            layer = tf.layers.max_pooling2d(layer,pool_size=[2,2],strides=2,name='c_mediastinum_2')
            layer = tf.layers.conv2d(layer,128,[3,3],padding="same",activation=tf.nn.relu,name='c_mediastinum_3')
            layer = tf.layers.max_pooling2d(layer,pool_size=[2,2],strides=2,name='c_mediastinum_4')

            layer = tf.layers.conv2d(layer,256,[3,3],padding="same",activation=tf.nn.relu,name='c_mediastinum_5')
            layer = tf.layers.max_pooling2d(layer,pool_size=[2,2],strides=2,name='c_mediastinum_6')
            layer = tf.reshape(layer, [-1, 16 * 16 * 256])

            layer = tf.layers.dense(layer,100,activation=tf.nn.relu,name='c_mediastinum_7')
            layer = tf.layers.dropout(layer,0.5,name='c_mediastinum_8')
            layer = tf.layers.dense(layer,5,activation=tf.nn.relu,name='c_mediastinum_9')

            logits = tf.nn.softmax(layer)
            return layer, logits

# classify for malignancy level of lung window
    def classify_lung_mali(self, image, y=None, reuse=False):
        with tf.variable_scope("classify_lung_mali") as scope:
            print('classify_lung_mali:')
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            layer = tf.layers.conv2d(image,64,[3,3],padding="same",activation=tf.nn.relu,name='c_classify_mali_1')
            layer = tf.layers.max_pooling2d(layer,pool_size=[2,2],strides=2,name='c_classify_mali_2')
            layer = tf.layers.conv2d(layer,128,[3,3],padding="same",activation=tf.nn.relu,name='c_classify_mali_3')
            layer = tf.layers.max_pooling2d(layer,pool_size=[2,2],strides=2,name='c_classify_mali_4')

            layer = tf.layers.conv2d(layer,256,[3,3],padding="same",activation=tf.nn.relu,name='c_classify_mali_5')
            layer = tf.layers.max_pooling2d(layer,pool_size=[2,2],strides=2,name='c_classify_mali_6')
            layer = tf.reshape(layer, [-1, 16 * 16 * 256])

            layer = tf.layers.dense(layer,100,activation=tf.nn.relu,name='c_classify_mali_7')
            layer = tf.layers.dropout(layer,0.5,name='c_classify_mali_8')
            layer = tf.layers.dense(layer,5,activation=tf.nn.relu,name='c_classify_mali_9')
            logits = tf.nn.softmax(layer)
            return layer, logits

    def classify_mediastinum_mali(self, image, y=None, reuse=False):
        with tf.variable_scope("classify_mediastinum_mali") as scope:
            print('classify_mediastinum_mali:')
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            layer = tf.layers.conv2d(image,64,[3,3],padding="same",activation=tf.nn.relu,name='c_mediastinum_mali_1')
            layer = tf.layers.max_pooling2d(layer,pool_size=[2,2],strides=2,name='c_mediastinum_mali_2')
            layer = tf.layers.conv2d(layer,128,[3,3],padding="same",activation=tf.nn.relu,name='c_mediastinum_mali_3')
            layer = tf.layers.max_pooling2d(layer,pool_size=[2,2],strides=2,name='c_mediastinum_mali_4')

            layer = tf.layers.conv2d(layer,256,[3,3],padding="same",activation=tf.nn.relu,name='c_mediastinum_mali_5')
            layer = tf.layers.max_pooling2d(layer,pool_size=[2,2],strides=2,name='c_mediastinum_mali_6')
            layer = tf.reshape(layer, [-1, 16 * 16 * 256])

            layer = tf.layers.dense(layer,100,activation=tf.nn.relu,name='c_mediastinum_mali_7')
            layer = tf.layers.dropout(layer,0.5,name='c_mediastinum_mali_8')
            layer = tf.layers.dense(layer,5,activation=tf.nn.relu,name='c_mediastinum_mali_9')

            logits = tf.nn.softmax(layer)
            return layer, logits