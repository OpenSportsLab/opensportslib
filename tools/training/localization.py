import argparse

from opensportslib.apis import LocalizationModel


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal localization training script.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--weights", default=None, help="Path to pretrained weights (optional).")
    parser.add_argument("--train-set", required=True, help="Path to train annotations JSON.")
    parser.add_argument("--valid-set", required=True, help="Path to validation annotations JSON.")
    parser.add_argument("--test-set", required=True, help="Path to test annotations JSON.")
    return parser.parse_args()


def main():
    args = parse_args()

    my_model = LocalizationModel(
        config=args.config,
        weights=args.weights,
    )

    my_model.train(
        train_set=args.train_set,
        valid_set=args.valid_set,
    )

    predictions = my_model.infer(
        test_set=args.test_set,
    )

    print(predictions)

    metrics = my_model.evaluate(
        test_set=args.test_set,
    )

    print(metrics)


if __name__ == "__main__":
    main()
