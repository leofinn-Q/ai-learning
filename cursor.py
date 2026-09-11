from pathlib import Path

FILE_PATH = Path(r"C:\Users\lyf88\Desktop\歌词\王菲 我爱你.txt")
OUTPUT_PATH = Path(r"C:\Users\lyf88\Desktop\歌词\词频统计结果.txt")
TARGET_WORD = "爱情"


def main() -> None:
    text = FILE_PATH.read_text(encoding="utf-8")
    count = text.count(TARGET_WORD)
    result = f"{TARGET_WORD}  {count}\n"

    OUTPUT_PATH.write_text(result, encoding="utf-8")
    print(result, end="")
    print(f"结果已保存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
