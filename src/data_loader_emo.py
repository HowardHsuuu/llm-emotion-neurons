import os
from datasets import load_dataset

def load_emotion_splits(emotion: str, split: str = 'train'):
    # split（train/validation/test）
    ds = load_dataset("go_emotions", split=split)

    label_names = ds.features["labels"].feature.names
    emo_idx     = label_names.index(emotion)
    neu_idx     = label_names.index("neutral")

    def is_pure_emotion(example):
        return example["labels"] == [emo_idx]
    def is_pure_neutral(example):
        return example["labels"] == [neu_idx]

    emo_ds    = ds.filter(is_pure_emotion)
    neutral_ds = ds.filter(is_pure_neutral)

    return emo_ds["text"], neutral_ds["text"]


def save_lists(texts, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line.replace("\n", " ") + "\n")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", type=str, required=True,
                        help="要篩選的情緒名稱（e.g. anger, joy...）")
    parser.add_argument("--split",   type=str, default="train",
                        help="GoEmotions 資料集切分 (train/validation/test)")
    args = parser.parse_args()

    emo_texts, neu_texts = load_emotion_splits(args.emotion, args.split)

    # data/processed/{emotion}/
    save_lists(emo_texts, f"data/processed/{args.emotion}/{args.split}_emotion.txt")
    save_lists(neu_texts, f"data/processed/{args.emotion}/{args.split}_neutral.txt")

    print(f"Saved {len(emo_texts)} '{args.emotion}' samples and {len(neu_texts)} neutral samples.")

if __name__ == '__main__':
    main()
