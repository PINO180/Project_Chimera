# remove_decorator.py - 全ての@handle_zero_stdデコレータを削除
import re

INPUT_FILE = 'independent_features.py'
OUTPUT_FILE = 'independent_features_clean.py'

def main():
    print(f"'{INPUT_FILE}' から全ての@handle_zero_stdデコレータを削除しています...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"エラー: '{INPUT_FILE}' が見つかりません。")
        return

    removed_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        i = 0
        while i < len(lines):
            line = lines[i]
            # @handle_zero_stdデコレータの行かチェック
            if '@handle_zero_std' in line.strip():
                print(f"  -> {line.strip()} を削除しました。")
                removed_count += 1
                i += 1  # デコレータ行をスキップ
                continue
            
            f_out.write(line)
            i += 1

    print(f"\n処理が完了しました！ {removed_count}個のデコレータを削除し、結果を'{OUTPUT_FILE}'に保存しました。")

if __name__ == '__main__':
    main()