                        # 直接パーティションディレクトリから読み込み（確実な方法）
                        partition_path = Path(input_path) / f"timeframe={tf}"
                        
                        if not partition_path.exists():
                            logger.warning(f"パーティション未存在: {partition_path}")
                            continue
                        
                        # パーティション内のParquetファイルを探す
                        parquet_files = list(partition_path.glob("*.parquet"))
                        
                        if not parquet_files:
                            logger.warning(f"時間軸{tf}: Parquetファイルなし")
                            continue
                        
                        # 空でないファイルを探して結合
                        tf_data_chunks = []
                        files_read = 0
                        target_rows = 10000  # 目標行数
                        
                        for pfile in parquet_files:
                            if files_read >= 10:  # 最大10ファイルまで
                                break
                            
                            try:
                                chunk = cudf.read_parquet(pfile)
                                if len(chunk) > 0:
                                    tf_data_chunks.append(chunk)
                                    files_read += 1
                                    
                                    # 十分なデータが集まったら停止
                                    total_rows = sum(len(c) for c in tf_data_chunks)
                                    if total_rows >= target_rows:
                                        break
                                        
                            except Exception as file_error:
                                logger.debug(f"ファイル{pfile}読み込みエラー: {file_error}")
                                continue
                        
                        if tf_data_chunks:
                            # データフレームを結合
                            tf_data = cudf.concat(tf_data_chunks, ignore_index=True)
                            
                            # 目標行数に制限
                            if len(tf_data) > target_rows:
                                tf_data = tf_data.head(target_rows)
                            
                            # 必要な列のみ選択
                            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                            available_cols = [col for col in required_cols if col in tf_data.columns]
                            
                            if available_cols:
                                tf_data = tf_data[available_cols]
                                
                                # timestampをdatetimeにリネーム
                                if 'timestamp' in tf_data.columns:
                                    tf_data = tf_data.rename(columns={'timestamp': 'datetime'})
                                
                                logger.info(f"時間軸{tf}: {files_read}ファイルから{len(tf_data)}行読み込み成功")
                            else:
                                logger.warning(f"時間軸{tf}: 必要な列が見つからない")
                                continue
                        else:
                            logger.warning(f"時間軸{tf}: 読み込み可能なファイルなし")
                            continue
                            
                    except Exception as read_error:
                        logger.error(f"時間軸{tf}: 読み込みエラー - {read_error}")
                        continue