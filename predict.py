
from torch import optim
from torch import nn

from lib.dataload import write_labels
from lib.checkpoint import load_checkpoint
from lib.args import get_predict_args
from lib.infer import output

def main():
    # Get command line arguments
    in_args = get_predict_args()

    # Load and process label mapping
    label_map = write_labels(in_args.labels)

    # Load and process model
    model = load_checkpoint(in_args.checkpoint, in_args.arch, in_args.device)

    # Make prediction and display results
    output(in_args.image, label_map, model)


if __name__ == '__main__':
    main()