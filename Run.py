from Train import run_training, count_parameters, plot_losses
import logging
import argparse


def choose_workload(args):
    if args.run == "train":
        run_training()
    elif args.run == "count_parameters":
        count_parameters()
    elif args.run == "plot_losses":
        plot_losses()


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Script to run neural network training"
    )
    parser.add_argument("run", default="train", help="What to execute")

    choose_workload(parser.parse_args())


if __name__ == "__main__":
    main()
