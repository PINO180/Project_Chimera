#!/usr/bin/env python3
"""
CSVファイルの詳細フォーマット確認用デバッグスクリプト
"""

def debug_csv_format():
    input_file = r"C:\clonecloneclone\data\XAUUSDm_row.csv"
    
    print("=== CSVフォーマット詳細分析 ===")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        print("最初の10行:")
        for i in range(10):
            line = f.readline().strip()
            print(f"行{i+1}: [{repr(line)}]")
            
            if i == 0:
                print(f"  ヘッダー文字数: {len(line)}")
                print(f"  タブ数: {line.count(chr(9))}")
                print(f"  カンマ数: {line.count(',')}")
                print(f"  セミコロン数: {line.count(';')}")
                print(f"  パイプ数: {line.count('|')}")
            
            if i > 0 and line:
                # タブ分割を試す
                parts = line.split('\t')
                print(f"  タブ分割結果: {len(parts)} 部分")
                for j, part in enumerate(parts[:7]):  # 最初の7部分のみ表示
                    print(f"    部分{j}: [{repr(part)}]")
                
                # カンマ分割も試す
                parts_comma = line.split(',')
                if len(parts_comma) > 1:
                    print(f"  カンマ分割結果: {len(parts_comma)} 部分")
                    for j, part in enumerate(parts_comma[:7]):
                        print(f"    部分{j}: [{repr(part)}]")
                
                break
    
    print("\n=== 文字エンコーディング確認 ===")
    # 文字コード確認
    try:
        with open(input_file, 'rb') as f:
            raw_bytes = f.read(1000)
            print(f"最初の1000バイト（16進）: {raw_bytes[:100].hex()}")
            
            # UTF-8として読み取り
            try:
                decoded_utf8 = raw_bytes.decode('utf-8')
                print("UTF-8デコード: 成功")
            except:
                print("UTF-8デコード: 失敗")
                
            # Shift_JISとして読み取り
            try:
                decoded_sjis = raw_bytes.decode('shift_jis')
                print("Shift_JISデコード: 成功")
            except:
                print("Shift_JISデコード: 失敗")
                
            # CP1252として読み取り
            try:
                decoded_cp1252 = raw_bytes.decode('cp1252')
                print("CP1252デコード: 成功")
            except:
                print("CP1252デコード: 失敗")
                
    except Exception as e:
        print(f"バイナリ読み取りエラー: {e}")
    
    print("\n=== ファイル統計 ===")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            line_count = 0
            empty_lines = 0
            total_chars = 0
            
            for line in f:
                line_count += 1
                if line.strip() == "":
                    empty_lines += 1
                total_chars += len(line)
                
                if line_count >= 1000:  # 最初の1000行で統計
                    break
            
            print(f"最初の{line_count}行の統計:")
            print(f"  空行数: {empty_lines}")
            print(f"  平均文字数: {total_chars/line_count:.1f}")
            
    except Exception as e:
        print(f"統計取得エラー: {e}")

if __name__ == "__main__":
    debug_csv_format()