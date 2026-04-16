import collections
from pathlib import Path


def sort_and_save_features_by_timeframe(input_path: Path, output_path: Path):
    """
    特徴量リストファイルを読み込み、タイムフレームごとに並べ替えて集計し、
    結果を指定されたファイルパスに保存する。

    Args:
        input_path (Path): 読み込む特徴量リストのファイルパス。
        output_path (Path): 結果を保存するファイルパス。
    """
    # タイムフレームの論理的な順序を定義
    timeframe_order = [
        "tick",
        "M0.5",
        "M1",
        "M3",
        "M5",
        "M8",
        "M15",
        "M30",
        "H1",
        "H4",
        "H6",
        "H12",
        "D1",
        "W1",
        "MN",
    ]

    # 特徴量をタイムフレームごとに格納する辞書
    features_by_timeframe = collections.defaultdict(list)

    try:
        # 入力ファイルを読み込む
        with input_path.open("r", encoding="utf-8") as f:
            features = [line.strip() for line in f if line.strip()]

        for feature in features:
            try:
                suffix = feature.rsplit("_", 1)[1]
                if suffix in timeframe_order:
                    features_by_timeframe[suffix].append(feature)
            except IndexError:
                # 処理できない行はスキップ
                continue

    except FileNotFoundError:
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        return
    except Exception as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {e}")
        return

    try:
        # --- 結果をファイルに出力 ---
        # 出力先ディレクトリが存在しない場合は作成
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f_out:
            # 1. 各タイムフレームの合計数を書き込む
            f_out.write("--- 特徴量のタイムフレーム別合計数 ---\n")
            total_features = 0

            for tf in timeframe_order:
                if tf in features_by_timeframe:
                    count = len(features_by_timeframe[tf])
                    f_out.write(f"{tf.ljust(5)}: {count} 個\n")
                    total_features += count

            f_out.write("------------------------------------\n")
            f_out.write(f"合計特徴量数: {total_features} 個\n\n")

            # 2. タイムフレームごとにソートされた特徴量リストを書き込む
            f_out.write("--- タイムフレーム別ソート済み特徴量リスト ---\n")
            for timeframe in timeframe_order:
                if timeframe in features_by_timeframe:
                    # 各リスト内をアルファベット順にソート
                    sorted_features = sorted(features_by_timeframe[timeframe])
                    f_out.write(f"\n# --- Timeframe: {timeframe} ---\n")
                    for feature in sorted_features:
                        f_out.write(f"{feature}\n")

        # 完了メッセージをコンソールに表示
        print(f"✅ 処理が完了しました。結果は以下のファイルに保存されました。")
        print(output_path)

    except Exception as e:
        print(f"ファイルの書き込み中にエラーが発生しました: {e}")


if __name__ == "__main__":
    # 入力ファイルパス
    input_file_path = Path(
        "/workspace/data/XAUUSD/stratum_3_artifacts/1A_2B/final_survivor_feature_list.txt"
    )

    # 指定された出力ディレクトリとファイル名
    output_dir = Path("/workspace/data/XAUUSD/stratum_3_artifacts/deep_research_data")
    output_file_name = "sorted_and_counted_features.txt"
    output_file_path = output_dir / output_file_name

    # 関数を実行
    sort_and_save_features_by_timeframe(input_file_path, output_file_path)
