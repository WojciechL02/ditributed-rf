import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model-input", type=str, required=False, default="model")
    parser.add_argument("query-input", type=str, required=False, default="queries")
    parser.add_argument("predictions-output", type=str, required=False, default="results")
    args = parser.parse_args()

if __name__ == "__main__":
    main()
