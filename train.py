import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset-path", type=str, required=True)
    parser.add_argument("model-output", type=str, required=False, default="model")
    parser.add_argument("n-trees", type=int, required=True)
    parser.add_argument("seed", type=int, required=False, default=42)
    args = parser.parse_args()

if __name__ == "__main__":
    main()
