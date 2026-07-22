import argparse

from opensportslib.apis import VQAModel


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal VQA training script.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--weights", default=None, help="Optional pretrained weights or adapter path.")
    parser.add_argument(
        "--train-set",
        default=None,
        help="Path to train annotations JSON. Defaults to the config train split.",
    )
    parser.add_argument(
        "--valid-set",
        default=None,
        help="Path to validation annotations JSON. Defaults to the config valid split.",
    )
    parser.add_argument(
        "--test-set",
        default=None,
        help="Path to test annotations JSON. Defaults to the config test split.",
    )
    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Train only and skip the post-training inference/evaluation pass.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    my_model = VQAModel(
        config=args.config,
        weights=args.weights,
    )

    best_ckpt = my_model.train(
        train_set=args.train_set,
        valid_set=args.valid_set,
        use_wandb=args.use_wandb,
    )
    print(best_ckpt)

    if args.skip_infer:
        return

    predictions = my_model.infer(
        test_set=args.test_set,
        weights=best_ckpt,
        use_wandb=args.use_wandb,
    )
    print(predictions)

    metrics = my_model.evaluate(
        test_set=args.test_set,
        predictions=predictions,
        weights=best_ckpt,
        use_wandb=args.use_wandb,
    )
    print(metrics)


if __name__ == "__main__":
    main()
