# 金融tickデータを日付指定して整形するスクリプト。
# 使用時はこのスクリプトをローカルのDownloadsフォルダに移してから以下のコマンドでパワーシェルから実行すること。
# python C:\Users\Public\Documents\Downloads\s1_0_X_filter.py
# また入力・出力パス・日付も使用時に変更してください。


# 新しいファイルの最初（先頭）の10行を確認するコマンド
# Get-Content "C:\Users\Public\Documents\Downloads\XAUUSDm_custom_filtered.csv" -TotalCount 10
# 新しいファイルの終わり（末尾）の10行を確認するコマンド
# Get-Content "C:\Users\Public\Documents\Downloads\XAUUSDm_custom_filtered.csv" -Tail 10
# 元のファイルの最初（先頭）の10行を確認するコマンド
# Get-Content "C:\Users\Public\Documents\Downloads\XAUUSDm_202107021340_202603232359.csv" -TotalCount 10
# 元のファイルの終わり（末尾）の10行を確認するコマンド
# Get-Content "C:\Users\Public\Documents\Downloads\XAUUSDm_202107021340_202603232359.csv" -Tail 10

import time

# 入出力ファイルのパス（適宜ファイル名を変更してください）
input_file = (
    r"C:\Users\Public\Documents\Downloads\XAUUSDm_202107021340_202603232359.csv"
)
output_file = r"C:\Users\Public\Documents\Downloads\XAUUSDm_custom_filtered.csv"

# 抽出する期間（この日付「を含む」期間が抽出されます）
start_date = "2021.07.12"
end_date = "2026.03.23"  # "2026.03.23"に設定した場合 2026.03.23 23:59:59 までの全ティックが含まれます

print(f"処理を開始します。{start_date} から {end_date} までのデータを抽出します...")
start_time = time.time()

with (
    open(input_file, "r", encoding="utf-8", errors="ignore") as f_in,
    open(output_file, "w", encoding="utf-8") as f_out,
):
    # ヘッダー（1行目）を書き出す
    header = f_in.readline()
    f_out.write(header)

    count_read = 0
    count_written = 0

    for line in f_in:
        count_read += 1

        # 行の先頭10文字（日付部分）を比較
        date_str = line[:10]
        if start_date <= date_str <= end_date:
            f_out.write(line)
            count_written += 1

        # 1000万行ごとに進捗を表示
        if count_read % 10000000 == 0:
            elapsed = time.time() - start_time
            print(
                f"{count_read:,} 行処理完了... (抽出: {count_written:,} 行) 経過時間: {elapsed:.1f}秒"
            )

total_time = time.time() - start_time
print("\n処理が完了しました！")
print(f"総読み込み行数: {count_read:,} 行")
print(f"総書き出し行数: {count_written:,} 行")
print(f"合計所要時間: {total_time:.1f} 秒")
