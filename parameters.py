import argparse

from getpass import getuser

user = getuser()


def print_args(args, get_str=False):
    if "delimiter" in args:
        delimiter = args.delimiter
    elif "sep" in args:
        delimiter = args.sep
    else:
        delimiter = ";"
    print("###################################################################")
    print("args: ")
    keys = sorted(
        [
            a
            for a in dir(args)
            if not (
                a.startswith("__")
                or a.startswith("_")
                or a == "sep"
                or a == "delimiter"
        )
        ]
    )
    values = [getattr(args, key) for key in keys]
    if get_str:
        keys_str = delimiter.join([str(a) for a in keys])
        values_str = delimiter.join([str(a) for a in values])
        print(keys_str)
        print(values_str)
        return keys_str, values_str
    else:
        for key, value in zip(keys, values):
            print(key, ": ", value, flush=True)
    print("ARGS FINISHED", flush=True)
    print("######################################################")


def get_args():
    parser = argparse.ArgumentParser(
        description='Pytorch Detecting Out-of-distribution examples in '
                    'neural networks')

    # Device arguments
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu index')

    # Model loading arguments
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--model_dir', default='./models', type=str,
                        help='model name for saving')

    # Architecture arguments
    parser.add_argument(
        '--architecture', default='densenet', type=str,
        help='underlying architecture (densenet | resnet | wideresnet)')
    parser.add_argument(
        '--similarity', default='cosine', type=str,
        help='similarity function for decomposed confidence '
             'numerator (cosine | inner | euclid | baseline | none)')
    parser.add_argument('--loss_type', default='ce', type=str)

    # Data loading arguments
    parser.add_argument('--data_dir', default=f'/home/{user}/data', type=str)
    parser.add_argument('--in_dataset',
                        default='CIFAR10',
                        type=str,
                        help='in-distribution dataset')
    parser.add_argument('--out_dataset',
                        # default='Imagenet',
                        default='SVHN',
                        type=str,
                        help='out-of-distribution dataset')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')

    # Training arguments
    parser.add_argument('--no_train', action='store_false', dest='train')
    parser.add_argument('--weight_decay', default=0.0001, type=float,
                        help='weight decay during training')
    parser.add_argument('--epochs', default=300, type=int,
                        help='number of epochs during training')

    # Testing arguments
    parser.add_argument('--no_test', action='store_false', dest='test')
    parser.add_argument('--magnitudes', nargs='+',
                        default=[0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08],
                        type=float,
                        help='perturbation magnitudes')

    parser.set_defaults(argument=True)
    args = parser.parse_args()
    print_args(args=args)
    return args
