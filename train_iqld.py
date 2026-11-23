"""Public training entry for the independent Role-IQLD method."""

from .train_iqld_expert import parse_args, train, update_iqld

__all__ = ["parse_args", "train", "update_iqld"]


if __name__ == "__main__":
    train(parse_args())
