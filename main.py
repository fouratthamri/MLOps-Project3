import argparse
from src import basic_cleaning, train_test_model, check_score
import logging


def go(args):
    """
    Run the pipeline
    """
    logging.basicConfig(level=logging.INFO)

    if args.action == "all" or args.action == "basic_cleaning":
        logging.info("Basic cleaning procedure started")
        basic_cleaning.main()

    if args.action == "all" or args.action == "train_test_model":
        logging.info("Train/Test model procedure started")
        train_test_model.main()

    if args.action == "all" or args.action == "check_score":
        logging.info("Score check procedure started")
        check_score.main()



if __name__ == "__main__":
    """
    Main entrypoint
    """
    parser = argparse.ArgumentParser(description="Classifier training pipeline")

    parser.add_argument(
        "--action",
        type=str,
        choices=["clean_data",
                 "train_test_model",
                 "evaluate_slice_performance",
                 "all"],
        default="all",
        help="Pipeline step to perform"
    )

    args = parser.parse_args()

    go(args)