# Put the code for your API here.
import argparse
from src import clean_step, train_step, slice_eval
import logging


def go(args):
    """
    Execute the pipeline
    """
    logging.basicConfig(level=logging.INFO)

    if args.action == "all" or args.action == "clean_data":
        logging.info("clean data")
        clean_step.clean_data()

    if args.action == "all" or args.action == "train_test_model":
        logging.info("train and test rf model")
        train_step.main()

    if args.action == "all" or args.action == "evaluate_slice_performance":
        logging.info("get model metrics for specific data slices")
        slice_eval.main()


if __name__ == "__main__":
    """
    Main entrypoint
    """
    parser = argparse.ArgumentParser(description="ML training pipeline")

    parser.add_argument(
        "--action",
        type=str,
        choices=["clean_data",
                 "train_test_model",
                 "evaluate_slice_performance",
                 "all"],
        default="all",
        help="Pipeline action"
    )

    args = parser.parse_args()

    go(args)