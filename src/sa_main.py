import os
import tensorflow as tf
from sa_solver import Solver
from sa_convert import Conversion

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('train_interval', 1, 'training interval between discriminator and generator, default: 1')
tf.flags.DEFINE_integer('ratio_gan2seg', 20, 'ratio of gan loss to seg loss, default: 10')
tf.flags.DEFINE_string('discriminator', 'image', 'type of discriminator [pixel|patch1|patch2|image],default: image')
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 2e-4')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of adam, default: 0.5')
tf.flags.DEFINE_integer('iters', 50000, 'number of iteratons, default: 50000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency, default: 100')
tf.flags.DEFINE_integer('eval_freq', 500, 'evaluation frequency, default: 500')
tf.flags.DEFINE_integer('sample_freq', 200, 'sample frequency, default: 200')
tf.flags.DEFINE_integer('save_freq', 4000, 'save model frequency, default: 4000')
tf.flags.DEFINE_integer('lambda1', 100, 'ratio seg loss')
tf.flags.DEFINE_integer('lambda2', 1, 'ratio gan loss')

tf.flags.DEFINE_string('gpu_index', '2', 'gpu index, default: 0')

tf.flags.DEFINE_bool('is_test', False, 'default: False (train)')

tf.flags.DEFINE_string('test_dir','./test','test input file dir')
tf.flags.DEFINE_string('model_dir','./model','model checkpoint file dir')
tf.flags.DEFINE_string('output_dir','./output_dir','test output dir')

tf.flags.DEFINE_bool('is_convert',False,'is_convert')

tf.flags.DEFINE_integer('mn',0, 'all model checkpoint paths index')
tf.flags.DEFINE_integer('fn',4000, 'test_output_folder_name')
tf.flags.DEFINE_bool('is_single','False','is_single')
tf.flags.DEFINE_string('input_file','input_file.png','if is_single==True, convert_input_file_dir')

def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    #solver = Solver(FLAGS)
    #conversion= Conversion(FLAGS)
    if FLAGS.is_convert:
        conversion= Conversion(FLAGS)
        conversion.convert()

    elif not FLAGS.is_convert:
        solver = Solver(FLAGS)
        if FLAGS.is_test:
            solver.test()
        if not FLAGS.is_test:
            solver.train()


if __name__ == '__main__':
    tf.app.run()
