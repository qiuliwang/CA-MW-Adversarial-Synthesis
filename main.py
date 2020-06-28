from model import MCGAN
import tensorflow as tf
from utils import LIDC
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags

flags.DEFINE_integer("output_size", 128 , "the size of generate image")
flags.DEFINE_float("learn_rate", 0.0002, "the learning rate for gan")
flags.DEFINE_integer("batch_size", 64, "the batch number")
flags.DEFINE_integer("z_dim", 100, "the dimension of noise z")
flags.DEFINE_integer("y_dim", 15, "the dimension of condition y")
flags.DEFINE_string("log_dir" , "/tmp/tensorflow_mnist" , "the path of tensorflow's log")
flags.DEFINE_string("model_path" , "model/model.ckpt" , "the path of model")
flags.DEFINE_string("visua_path" , "visualization" , "the path of visuzation images")
flags.DEFINE_integer("op" , 0, "0: train ; 1:test ; 2:visualize")

FLAGS = flags.FLAGS

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='facades', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200000, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=500, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')

args = parser.parse_args()

#
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)
if not os.path.exists(FLAGS.model_path):
    os.makedirs(FLAGS.model_path)

def main(_):

    mn_object = LIDC()

    cg = MCGAN(data_ob = mn_object, output_size=FLAGS.output_size, learn_rate=FLAGS.learn_rate
         , batch_size=FLAGS.batch_size, z_dim=FLAGS.z_dim, y_dim=FLAGS.y_dim, log_dir=FLAGS.log_dir
         , model_path=FLAGS.model_path, load = False)

    cg.build_model()
    
    cg.train(args)

    if FLAGS.op == 0:

        cg.train(args)

    elif FLAGS.op == 1:

        cg.test()

if __name__ == '__main__':
    tf.app.run()
