"""Filter an expert dataset by episode quality metrics."""

import argparse
from pathlib import Path

import numpy as np
import torch


def episode_total_reward(episode):
    return float(sum(t[4].sum() for t in episode["transitions"]))


def episode_length(episode):
    return len(episode["transitions"])


def filter_dataset(input_path: str, output_path: str, top_k: int,
                   metric: str = "reward"):
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    episodes = data["episodes"]
    if metric == "reward":
        scores = [episode_total_reward(ep) for ep in episodes]
        descending = True
    elif metric == "length":
        scores = [episode_length(ep) for ep in episodes]
        descending = False
    elif metric == "separation":
        scores = [ep["minimum_separation"] for ep in episodes]
        descending = True
    else:
        raise ValueError(f"unknown metric: {metric}")

    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    selected = [episodes[i] for i in order[:top_k]]

    new_data = {
        **data,
        "episodes": selected,
        "source": f"{data.get('source', 'unknown')}_filtered_{metric}_top{top_k}",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_data, output_path)
    print(
        f"selected {len(selected)} episodes from {len(episodes)} "
        f"by {metric} (descending={descending}) -> {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument(
        "--metric", choices=("reward", "length", "separation"),
        default="reward",
    )
    args = parser.parse_args()
    filter_dataset(args.input, args.output, args.top_k, args.metric)


if __name__ == "__main__":
    main()
