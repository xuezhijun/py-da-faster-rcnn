import argparse
from model.utils.config import cfg, cfg_from_file, cfg_from_list


def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')

    parser.add_argument('--dataset', dest='dataset',
                        help='source training dataset',
                        default='sim10k', type=str)
    parser.add_argument('--dataset_t', dest='dataset_t',
                        help='target training dataset',
                        default='cityscape_car', type=str)
    parser.add_argument('--net', dest='net',
                        help='backbone: vgg16, res101, res50',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--max_epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=10, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models',
                        default="models", type=str)
    parser.add_argument('--load_name', dest='load_name',
                        help='path to load models',
                        default="models", type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--load_config', dest='config_I2I',
                        help='name of config file for loading I2I model',
                        default='cityscape.yaml', type=str)
    parser.add_argument('--load_model', dest='path_I2I',
                        help="provide the path for loading I2I model",
                        default='', type=str)
    ################################################################################
    parser.add_argument('--detach', dest='detach',
                        help='whether use detach',
                        action='store_false')
    parser.add_argument('--ef', dest='ef',
                        help='whether use exponential focal loss',
                        action='store_true')
    parser.add_argument('--lc', dest='lc',
                        help='whether use context vector for local/pixel level',
                        action='store_true')
    parser.add_argument('--gc', dest='gc',
                        help='whether use context vector for global level',
                        action='store_true')

    parser.add_argument('--gamma', dest='gamma',
                        help='value of gamma',
                        default=5.0, type=float)
    parser.add_argument('--eta', dest='eta',
                        help='trade-off parameter between detection loss and domain-alignment loss. Used for Car datasets',
                        default=0.1, type=float)

    parser.add_argument('--mode', dest='mode',
                        help='the mode of domain adaptation',
                        default='gcn_adapt', type=str)
    parser.add_argument('--rpn_mode', dest='rpn_mode',
                        help='the mode of domain adaptation for RPN',
                        default='gcn_adapt', type=str)

    parser.add_argument('--da_weight', dest='da_weight',
                        help='the weight of RCNN adaptation loss',
                        default=1.0, type=float)
    parser.add_argument('--rpn_da_weight', dest='rpn_da_weight',
                        help='the weight of RPN adaptation loss',
                        default=1.0, type=float)

    parser.add_argument('--cosine_rpn_da_weight', dest='cosine_rpn_da_weight',
                        help='cosine_rpn_da_weight',
                        action='store_true')
    parser.add_argument('--warm_up', dest='warm_up',
                        help='warm_up iters',
                        default=200, type=int)

    parser.add_argument('--pos_ratio', dest='pos_ratio',
                        help='ration of positive example',
                        default=0.25, type=float)
    parser.add_argument('--rpn_bs', dest='rpn_bs',
                        help='rpn batchsize',
                        default=128, type=int)
    parser.add_argument('--train_bs', dest='train_bs',
                        help='BATCH_SIZE',
                        default=128, type=int)

    ################################################################################
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--vis', dest='vis',
                        help='True if you wish to have visualisation',
                        action='store_true')
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_steps', dest='lr_decay_steps',
                        help='step to do learning rate decay, unit is epoch',
                        default='5', type=str)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)
    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        action='store_true')
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images/Img", type=str)
    parser.add_argument('--det_image_dir', dest='det_image_dir',
                        help='directory to save images for demo',
                        default="images/Det", type=str)
    parser.add_argument('--da_method', dest='da_method',
                        help='which da method to be used for demo',
                        default='DA', type=str)

    args = parser.parse_args()
    return args


def set_dataset_args(args, test=False):
    '''
    pascal -> watercolor         : pascal_voc_water -> water
    pascal -> clipart            : pascal_voc_0712 -> clipart
    cityscape -> foggy cityscape : cityscape -> foggy_cityscape
    sim10k -> cityscape          : sim10k -> cityscape_car
    SYNTHIA -> cityscape         : synthia -> cityscape_synthia
    kitti <-> cityscape          : kitti_car <-> cityscape_car
    Syn2Real -> COCO             : syn2real -> coco
    '''
    if not test:
        ######################## source training dataset: --dataset=='str' --> imdb_name
        if args.dataset == "pascal_voc":
            args.imdb_name = "voc_2007_trainval"  # "voc_2012_trainval"
            args.imdbval_name = "voc_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "pascal_voc_water":
            args.imdb_name = "voc_water_2007_trainval+voc_water_2012_trainval"  ###### pascal_voc_water ######
            args.imdbval_name = "voc_clipart_2007_trainval+voc_clipart_2012_trainval"
            args.imdb_name_cycle = "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "pascal_voc_cycleclipart":
            args.imdb_name = "voc_cycleclipart_2007_trainval+voc_cycleclipart_2012_trainval"
            args.imdbval_name = "voc_cycleclipart_2007_trainval+voc_cycleclipart_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "pascal_voc_cyclewater":
            args.imdb_name = "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.imdbval_name = "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "pascal_voc_0712":
            args.imdb_name = "voc_2007_trainval+voc_2012_trainval"  ###### pascal_voc_0712 ######
            args.imdbval_name = "voc_2007_test"
            args.imdb_name_cycle = "voc_cycleclipart_2007_trainval+voc_cycleclipart_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "foggy_cityscape":
            args.imdb_name = "foggy_cityscape_trainval"
            args.imdbval_name = "foggy_cityscape_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "vg":
            args.imdb_name = "vg_150-50-50_minitrain"
            args.imdbval_name = "vg_150-50-50_minival"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

        elif args.dataset == "cityscape":
            args.imdb_name = "cityscape_train"  ###### cityscape ######
            args.imdbval_name = "cityscape_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "sim10k":
            args.imdb_name = "sim10k_train"  ###### sim10k ######
            args.imdbval_name = "sim10k_train"
            args.imdb_name_cycle = "sim10k_cycle_train"  # "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "sim10k_cycle":
            args.imdb_name = "sim10k_cycle_train"
            args.imdbval_name = "sim10k_cycle_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "init_sunny":
            args.imdb_name = "init_sunny_trainval"
            args.imdbval_name = "init_sunny_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "init_night":
            args.imdb_name = "init_night_train"
            args.imdbval_name = "init_night_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "init_rainy":
            args.imdb_name = "init_rainy_train"
            args.imdbval_name = "init_rainy_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "init_cloudy":
            args.imdb_name = "init_cloudy_train"
            args.imdbval_name = "init_cloudy_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "kitti_car":
            args.imdb_name = "kitti_car_train"  ###### kitti_car ######
            args.imdbval_name = "kitti_car_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "cityscape_car":
            args.imdb_name = "cityscape_car_train"
            args.imdbval_name = "cityscape_car_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "cityscape_kitti":
            args.imdb_name = "cityscape_kitti_trainval"
            args.imdbval_name = "cityscape_kitti_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "synthia":
            args.imdb_name = "synthia_train"  ###### synthia ######
            args.imdbval_name = "synthia_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "coco":
            args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
            args.imdbval_name = "coco_2014_minival"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

        elif args.dataset == "imagenet":
            args.imdb_name = "imagenet_train"
            args.imdbval_name = "imagenet_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        ############### target training dataset: --dataset_t=='str' --> imdb_name_target
        if args.dataset_t == "water":
            args.imdb_name_target = "water_train"  ###### water ######
            args.imdbval_name_target = "water_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "clipart":
            args.imdb_name_target = "clipart_trainval"  ###### clipart ######
            args.imdbval_name_target = "clipart_test"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "cityscape":
            args.imdb_name_target = "cityscape_train"
            args.imdbval_name_target = "cityscape_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset_t == "cityscape_car":
            args.imdb_name_target = "cityscape_car_train"  ###### cityscape_car ######
            args.imdbval_name_target = "cityscape_car_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "kitti_car":
            args.imdb_name_target = "kitti_car_train"
            args.imdbval_name = "kitti_car_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset_t == "foggy_cityscape":
            args.imdb_name_target = "foggy_cityscape_train"  ###### foggy_cityscape ######
            args.imdbval_name_target = "foggy_cityscape_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset_t == "init_night":
            args.imdb_name_target = "init_night_train"
            args.imdbval_name_target = "init_night_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset_t == "init_rainy":
            args.imdb_name_target = "init_rainy_train"
            args.imdbval_name_target = "init_rainy_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset_t == "init_cloudy":
            args.imdb_name_target = "init_cloudy_train"
            args.imdbval_name_target = "init_cloudy_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset_t == "cityscape_synthia":
            args.imdb_name_target = "cityscape_synthia_train"  ###### cityscape_synthia ######
            args.imdbval_name_target = "cityscape_synthia_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']


    else:
        ################################ test dataset: --dataset=='str' --> imdbval_name
        if args.dataset == "pascal_voc":
            args.imdb_name = "voc_2007_trainval"
            args.imdbval_name = "voc_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

        elif args.dataset == "pascal_voc_0712":
            args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
            args.imdbval_name = "voc_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

        elif args.dataset == "sim10k":
            args.imdb_name = "sim10k_val"
            args.imdbval_name = "sim10k_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "cityscape":
            args.imdb_name = "cityscape_val"
            args.imdbval_name = "cityscape_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "kitti_car":
            args.imdb_name = "kitti_car_train"
            args.imdbval_name = "kitti_car_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "foggy_cityscape":
            args.imdb_name = "foggy_cityscape_test"
            args.imdbval_name = "foggy_cityscape_val"  ###### foggy_cityscape ######
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "cityscape_kitti":
            args.imdb_name = "cityscape_kitti_val"
            args.imdbval_name = "cityscape_kitti_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "water":
            args.imdb_name = "water_test"
            args.imdbval_name = "water_test"  ###### water ######
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "clipart":
            args.imdb_name = "clipart_test"
            args.imdbval_name = "clipart_test"  ###### clipart ######
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "cityscape_car":
            args.imdb_name = "cityscape_car_val"
            args.imdbval_name = "cityscape_car_val"  ###### cityscape_car ######
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "init_sunny":
            args.imdb_name = "init_sunny_trainval"
            args.imdbval_name = "init_sunny_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "init_night":
            args.imdb_name = "init_night_val"
            args.imdbval_name = "init_night_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "init_rainy":
            args.imdb_name = "init_rainy_val"
            args.imdbval_name = "init_rainy_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "init_cloudy":
            args.imdb_name = "init_cloudy_val"
            args.imdbval_name = "init_cloudy_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "cityscape_synthia":
            args.imdb_name = "cityscape_synthia_val"
            args.imdbval_name = "cityscape_synthia_val"  ###### cityscape_synthia ######
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    return args
