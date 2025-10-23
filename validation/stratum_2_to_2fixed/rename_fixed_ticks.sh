#!/bin/bash
# rename_fixed_ticks.sh

BASE_DIR="/workspace/data/XAUUSD/stratum_2_features_fixed"

cd "$BASE_DIR" || exit 1

for engine_dir in feature_value_*; do
    cd "$engine_dir" || continue
    
    # _fixedサフィックス付きディレクトリを検索
    for fixed_dir in *_tick_fixed; do
        if [ -d "$fixed_dir" ]; then
            # 元のディレクトリ名を取得（_fixedを除去）
            original_name="${fixed_dir%_fixed}"
            
            echo "処理中: $original_name"
            
            # 元のディレクトリが存在する場合はバックアップ
            if [ -d "$original_name" ]; then
                echo "  バックアップ: ${original_name}_backup"
                mv "$original_name" "${original_name}_backup"
            fi
            
            # _fixedを元の名前にリネーム
            echo "  リネーム: $fixed_dir → $original_name"
            mv "$fixed_dir" "$original_name"
        fi
    done
    
    cd ..
done

echo "完了！"