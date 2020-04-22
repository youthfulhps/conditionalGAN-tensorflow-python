import tensorflow as tf
import sa_Tensorflow_utils as tf_utils

def least_loss(y, y_hat):
    return tf.reduce_mean(tf.square(y - y_hat))

def pixel_loss(y, y_hat):
    return tf.reduce_mean(tf.abs(y-y_hat))

class CGAN(object):
    def __init__(self, sess, flags, image_size):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size

        self._gen_train_ops, self._dis_train_ops = [], []
        self.gen_c, self.dis_c = 64, 64

        self._build_net()       # initialize networks
        self._init_assign_op()  # initialize assign operations

        print('Initialized CGAN SUCCESS!\n')

    #initialize networks
    def _build_net(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.image_size[0], self.image_size[1],1], name='input')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.image_size[0], self.image_size[1],1], name='label')

        self.g_samples = self.generator(self.X)
        self.real_pair = tf.concat([self.X, self.Y], axis=3)
        self.fake_pair = tf.concat([self.X, self.g_samples], axis=3)

        self.d_real, self.d_logit_real = self.discriminator(self.real_pair)
        self.d_fake, self.d_logit_fake = self.discriminator(self.fake_pair, is_reuse=True)

        self.d_loss_real = least_loss(y=tf.ones_like(self.d_real), y_hat = self.d_logit_real)
        self.d_loss_fake = least_loss(y=tf.zeros_like(self.d_fake), y_hat = self.d_logit_fake)

        self.d_loss = (self.d_loss_real + self.d_loss_fake)/2


        self.gan_loss = least_loss(y=tf.ones_like(self.d_logit_fake), y_hat=self.d_logit_fake)/2
        self.seg_loss = pixel_loss(y=self.Y, y_hat=self.g_samples)

        self.g_loss =  self.flags.lambda2*self.gan_loss + self.flags.lambda1*self.seg_loss

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.dis_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1)\
            .minimize(self.d_loss, var_list=self.d_vars)
        self.dis_ops = [self.dis_op] + self._dis_train_ops
        self.dis_optim = tf.group(*self.dis_ops)

        self.gen_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1)\
            .minimize(self.g_loss, var_list=self.g_vars)
        self.gen_ops = [self.gen_op] + self._gen_train_ops
        self.gen_optim = tf.group(*self.gen_ops)

    #initialize assign operations
    def _init_assign_op(self):

        self.psnr_placeholder = tf.placeholder(tf.float32, name='psnr_placeholder')
        self.ssim_placeholder = tf.placeholder(tf.float32, name='ssim_placeholder')
        self.score_placeholder = tf.placeholder(tf.float32, name='score_best_placeholder')

        psnr = tf.Variable(0., trainable=False, dtype=tf.float32, name='psnr')
        ssim = tf.Variable(0., trainable=False, dtype=tf.float32, name='ssim')
        self.score = tf.Variable(0., trainable=False, dtype=tf.float32, name='score_best')

        self.score_assign_op = self.score.assign(self.score_placeholder)
        psnr_assign_op = psnr.assign(self.psnr_placeholder)
        ssim_assign_op = ssim.assign(self.ssim_placeholder)

        self.measure_assign_op = tf.group(psnr_assign_op, ssim_assign_op)
        self.model_out_dir = "{}/model_gan*{}+seg*{}".format(self.flags.output_dir, self.flags.lambda2 ,self.flags.lambda1)
        # for tensorboard
        if not self.flags.is_test:
            self.writer = tf.summary.FileWriter("{}/logs/gan*{}+seg*{}".format(self.flags.output_dir, self.flags.lambda2, self.flags.lambda1))


        psnr_summ = tf.summary.scalar("psnr_summary", psnr)
        ssim_summ = tf.summary.scalar("ssim_summary", ssim)
        score_summ = tf.summary.scalar("score_summary", self.score)

        self.g_loss_summary = tf.summary.scalar('generator_loss', self.g_loss)
        self.d_loss_summary = tf.summary.scalar('discriminator_loss', self.d_loss)

        self.measure_summary = tf.summary.merge([psnr_summ, ssim_summ, score_summ])

    #generator
    def generator(self, data, name='g_'):
        with tf.variable_scope(name):
            #(512,512,1) -> (512,512,64)
            conv1 = tf_utils.conv2d(data, self.gen_c, k_h=3, k_w=3,d_h=1, d_w=1, name='conv1_conv1')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu1')

            #(512,512,64) -> (256,256,128)
            conv2 = tf_utils.conv2d(conv1, 2*self.gen_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv2_conv1')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch1',_ops=self._gen_train_ops)
            conv2 = tf_utils.lrelu(conv2, name='conv2_lrelu1')

            #(256,256,128) -> (128,128,256)
            conv3 = tf_utils.conv2d(conv2, 4*self.gen_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv3_conv1')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch1',_ops=self._gen_train_ops)
            conv3 = tf_utils.lrelu(conv3, name='conv3_lrelu1')

            #(128,128,256) -> (128,128,256)
            res1 = tf_utils.residual_block(conv3, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name_prefix='residual_block_1_',_ops=self._gen_train_ops)
            res2 = tf_utils.residual_block(res1, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name_prefix='residual_block_2_',_ops=self._gen_train_ops)
            res3 = tf_utils.residual_block(res2, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name_prefix='residual_block_3_',_ops=self._gen_train_ops)
            res4 = tf_utils.residual_block(res3, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name_prefix='residual_block_4_',_ops=self._gen_train_ops)
            res5 = tf_utils.residual_block(res4, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name_prefix='residual_block_5_',_ops=self._gen_train_ops)
            res6 = tf_utils.residual_block(res5, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name_prefix='residual_block_6_',_ops=self._gen_train_ops)
            res7 = tf_utils.residual_block(res6, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name_prefix='residual_block_7_',_ops=self._gen_train_ops)
            res8 = tf_utils.residual_block(res7, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name_prefix='residual_block_8_',_ops=self._gen_train_ops)
            res9 = tf_utils.residual_block(res8, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name_prefix='residual_block_9_',_ops=self._gen_train_ops)

            #(128,128,256) -> (256,256,128) + conv2
            deconv1 = tf_utils.deconv2d(res9, [1,256,256,128], k_h=3, k_w=3, d_h=2, d_w=2, name='deconv1_deconv1')
            deconv1 = tf_utils.batch_norm(deconv1, name='deconv1_batch1', _ops=self._gen_train_ops)
            deconv1 = tf.nn.relu(deconv1, name='deconv1_relu1')
            deconv1 = tf.concat([deconv1, conv2], axis=3, name='deconv1_concat1')

            #(256,256,256) -> (512,512,64) + conv1
            deconv2 = tf_utils.deconv2d(deconv1, [1,512,512,64], k_h=3, k_w=3, d_h=2, d_w=2, name='deconv2_deconv1')
            deconv2 = tf_utils.batch_norm(deconv2, name='deconv2_batch1', _ops=self._gen_train_ops)
            deconv2 = tf.nn.relu(deconv2, name='deconv2_relu1')
            deconv2 = tf.concat([deconv2, conv1], axis=3, name='deconv2_concat1')

            deconv3 = tf_utils.deconv2d(deconv2, [1,512,512,1], k_h=1, k_w=1, d_h=1, d_w=1, name='deconv3_deconv1')
            deconv3 = tf.nn.tanh(deconv3, name='deconv3_tanh1')

        return deconv3



    #discriminator(pixel, patch1, patch2, image)
    def discriminator(self, data, is_reuse=False):
        if self.flags.discriminator == 'pixel':
            return self.discriminator_pixel(data, is_reuse=is_reuse)
        elif self.flags.discriminator == 'patch1':
            return self.discriminator_patch1(data, is_reuse=is_reuse)
        elif self.flags.discriminator == 'patch2':
            return self.discriminator_patch2(data, is_reuse=is_reuse)
        elif self.flags.discriminator == 'image':
            return self.discriminator_image(data, is_reuse=is_reuse)
        else:
            raise NotImplementedError

    def discriminator_pixel(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            # conv1: (N, 512, 512, 1) -> (N,, 512, 512, 64)
            conv1 = tf_utils.conv2d(data, self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv1')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu1')

            # conv2: (N, 512, 512, 64) -> (N, 512, 512, 128)
            conv2 = tf_utils.conv2d(conv1, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv1')
            conv2 = tf_utils.lrelu(conv2)

            # conv3: (N, 512, 512, 128) -> (N, 512, 512, 256)
            conv3 = tf_utils.conv2d(conv2, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
            conv3 = tf_utils.lrelu(conv3)

            # output layer: (N, 512, 512, 256) -> (N, 512, 512, 1)
            output = tf_utils.conv2d(conv3, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='conv_output')

            return tf.nn.sigmoid(output), output

    def discriminator_patch2(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            # conv1: (N, 512, 512, 1) -> (N,, 128, 128, 64)
            conv1 = tf_utils.conv2d(data, self.dis_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv1_conv1')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch1', _ops=self._dis_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu1')
            conv1 = tf_utils.conv2d(conv1, self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv2')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch2', _ops=self._dis_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu2')
            pool1 = tf_utils.max_pool_2x2(conv1, name='maxpool1')

            # conv2: (N, 128, 128, 64) -> (N, 128, 128, 128)
            conv2 = tf_utils.conv2d(pool1, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv1')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch1', _ops=self._dis_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu1')
            conv2 = tf_utils.conv2d(conv2, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv2')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch2', _ops=self._dis_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu2')
            pool2 = tf_utils.max_pool_2x2(conv2, name='maxpool2')

            # conv3: (N, 128, 128, 128) -> (N, 128, 128, 256)
            conv3 = tf_utils.conv2d(pool2, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch1', _ops=self._dis_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu1')
            conv3 = tf_utils.conv2d(conv3, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv2')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch2', _ops=self._dis_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu2')

            # output layer: (N, 128, 128, 256) -> (N, 128, 128, 1)
            output = tf_utils.conv2d(conv3, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='conv_output')

            return tf.nn.sigmoid(output), output

    def discriminator_patch1(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            # conv1: (N, 512, 512, 1) -> (N,, 128, 128, 64)
            conv1 = tf_utils.conv2d(data, self.dis_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv1_conv1')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch1', _ops=self._dis_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu1')
            conv1 = tf_utils.conv2d(conv1, self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv2')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch2', _ops=self._dis_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu2')
            pool1 = tf_utils.max_pool_2x2(conv1, name='maxpool1')

            # conv2: (N, 128, 128, 64) -> (N, 64, 64, 128)
            conv2 = tf_utils.conv2d(pool1, 2*self.dis_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv2_conv1')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch1', _ops=self._dis_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu1')
            conv2 = tf_utils.conv2d(conv2, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv2')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch2', _ops=self._dis_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu2')
            pool2 = tf_utils.max_pool_2x2(conv2, name='maxpool2')

            # conv3: (N, 64, 64, 128) -> (N, 32, 32, 256)
            conv3 = tf_utils.conv2d(pool2, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch1', _ops=self._dis_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu1')
            conv3 = tf_utils.conv2d(conv3, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv2')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch2', _ops=self._dis_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu2')
            pool3 = tf_utils.max_pool_2x2(conv3, name='maxpool3')

            # conv4: (N, 32, 32, 256) -> (N, 16, 16, 512)
            conv4 = tf_utils.conv2d(pool3, 8*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv1')
            conv4 = tf_utils.batch_norm(conv4, name='conv4_batch1', _ops=self._dis_train_ops)
            conv4 = tf.nn.relu(conv4, name='conv4_relu1')
            conv4 = tf_utils.conv2d(conv4, 8*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv2')
            conv4 = tf_utils.batch_norm(conv4, name='conv4_batch2', _ops=self._dis_train_ops)
            conv4 = tf.nn.relu(conv4, name='conv4_relu2')
            pool4 = tf_utils.max_pool_2x2(conv4, name='maxpool4')

            # conv5: (N, 16, 16, 512) -> (N, 16, 16, 1024)
            conv5 = tf_utils.conv2d(pool4, 16*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv1')
            conv5 = tf_utils.batch_norm(conv5, name='conv5_batch1', _ops=self._dis_train_ops)
            conv5 = tf.nn.relu(conv5, name='conv5_relu1')
            conv5 = tf_utils.conv2d(conv5, 16*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv2')
            conv5 = tf_utils.batch_norm(conv5, name='conv5_batch2', _ops=self._dis_train_ops)
            conv5 = tf.nn.relu(conv5, name='conv5_relu2')

            # output layer: (N, 16, 16, 1024) -> (N, 16, 16, 1)
            output = tf_utils.conv2d(conv5, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='conv_output')

            return tf.nn.sigmoid(output), output

    def discriminator_image(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            # conv1: (N, 512, 512, 1) -> (N,, 128, 128, 64)
            conv1 = tf_utils.conv2d(data, self.dis_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv1_conv1')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch1', _ops=self._dis_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu1')
            conv1 = tf_utils.conv2d(conv1, self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv2')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch2', _ops=self._dis_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu2')
            pool1 = tf_utils.max_pool_2x2(conv1, name='maxpool1')

            # conv2: (N, 128, 128, 64) -> (N, 32, 32, 128)
            conv2 = tf_utils.conv2d(pool1, 2*self.dis_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv2_conv1')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch1', _ops=self._dis_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu1')
            conv2 = tf_utils.conv2d(conv2, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv2')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch2', _ops=self._dis_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu2')
            pool2 = tf_utils.max_pool_2x2(conv2, name='maxpool2')

            # conv3: (N, 32, 32, 128) -> (N, 16, 16, 256)
            conv3 = tf_utils.conv2d(pool2, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch1', _ops=self._dis_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu1')
            conv3 = tf_utils.conv2d(conv3, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv2')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch2', _ops=self._dis_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu2')
            pool3 = tf_utils.max_pool_2x2(conv3, name='maxpool3')

            # conv4: (N, 16, 16, 256) -> (N, 8, 8, 512)
            conv4 = tf_utils.conv2d(pool3, 8*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv1')
            conv4 = tf_utils.batch_norm(conv4, name='conv4_batch1', _ops=self._dis_train_ops)
            conv4 = tf.nn.relu(conv4, name='conv4_relu1')
            conv4 = tf_utils.conv2d(conv4, 8*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv2')
            conv4 = tf_utils.batch_norm(conv4, name='conv4_batch2', _ops=self._dis_train_ops)
            conv4 = tf.nn.relu(conv4, name='conv4_relu2')
            pool4 = tf_utils.max_pool_2x2(conv4, name='maxpool4')

            # conv5: (N, 8, 8, 512) -> (N, 8, 8, 1024)
            conv5 = tf_utils.conv2d(pool4, 16*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv1')
            conv5 = tf_utils.batch_norm(conv5, name='conv5_batch1', _ops=self._dis_train_ops)
            conv5 = tf.nn.relu(conv5, name='conv5_relu1')
            conv5 = tf_utils.conv2d(conv5, 16*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv2')
            conv5 = tf_utils.batch_norm(conv5, name='conv5_batch2', _ops=self._dis_train_ops)
            conv5 = tf.nn.relu(conv5, name='conv5_relu2')

            # output layer: (N, 8, 8, 1024) -> (N, 1, 1, 1024) -> (N, 1)
            shape = conv5.get_shape().as_list()
            gap = tf.layers.average_pooling2d(inputs=conv5, pool_size=shape[1], strides=1, padding='VALID',
                                              name='global_vaerage_pool')
            gap_flatten = tf.reshape(gap, [-1, 16*self.dis_c])
            output = tf_utils.linear(gap_flatten, 1, name='linear_output')

            return tf.nn.sigmoid(output), output
    #train discriminator
    def train_dis(self, x_data, y_data, iter_time):

        feed_dict = {self.X: x_data, self.Y: y_data}
        # run discriminator
        _, d_loss, d_loss_summary = self.sess.run([self.dis_optim, self.d_loss, self.d_loss_summary], feed_dict=feed_dict)
        self.writer.add_summary(d_loss_summary, iter_time)

        return d_loss
    #train generator
    def train_gen(self, x_data, y_data, iter_time):
        feed_dict = {self.X: x_data, self.Y: y_data}
        # run generator
        _, g_loss, g_loss_summary = self.sess.run([self.gen_optim, self.g_loss, self.g_loss_summary], feed_dict=feed_dict)
        self.writer.add_summary(g_loss_summary, iter_time)

        return g_loss
    #measure assign
    def measure_assign(self,psnr, ssim, score, iter_time):
        feed_dict = {self.psnr_placeholder:psnr, self.ssim_placeholder:ssim}

        self.sess.run(self.measure_assign_op, feed_dict=feed_dict)

        summary = self.sess.run(self.measure_summary)
        self.writer.add_summary(summary, iter_time)
    #best score assign, Use it if you want to save the model when the score is the best.
    def best_score_assign(self, score):
        self.sess.run(self.score_assign_op, feed_dict={self.score_placeholder: score})
    #sample image
    def sample_imgs(self, x_data):
        return self.sess.run(self.g_samples, feed_dict={self.X: x_data})
