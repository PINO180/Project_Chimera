//+------------------------------------------------------------------+
//|                                         ProjectForgeReceiver.mq5 |
//|                        Copyright 2025, Project Forge Dev Team    |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Project Forge Dev Team"
#property link      "https://www.mql5.com"
#property version   "12.02"
#property strict

// 必要なライブラリ
#include <Trade\Trade.mqh>
#include <Zmq\Zmq.mqh>
#include <JAson.mqh>

// --- Inputs ---
input string   ControlEndpoint   = "tcp://*:5555"; // 制御・ハンドシェイク用 (REP)
input string   DataEndpoint      = "tcp://*:5556"; // バルクデータ転送用 (PUSH)
input string   HeartbeatEndpoint = "tcp://*:5558"; // ハートビート用 (REP)
input int      MagicNumber       = 20250101;
input double   MaxSlippage       = 3.0;

// --- Global Objects (ZMQ OOP) ---
Context *g_context = NULL;
Socket  *g_control_socket = NULL;   // REP: 5555
Socket  *g_data_socket = NULL;      // PUSH: 5556
Socket  *g_heartbeat_socket = NULL; // REP: 5558

// --- Data Transfer State Machine ---
MqlRates g_history_data[]; // 全履歴データキャッシュ
int      g_total_bars    = 0;
int      g_current_index = 0;
int      g_chunk_size    = 50000; // 1チャンクあたりの送信バー数
bool     g_is_sending    = false; // データ送信中フラグ

// --- Constants ---
const int MQLRATES_STRUCT_SIZE = 60;      // MqlRates構造体の固定サイズ (bytes)
const int DATA_REQUEST_COUNT   = 3000000; // 要求する履歴バー数 (300万行)

// --- Trading Object ---
CTrade trade;

// =================================================================
// [V12.1] M0.5 (30秒足) 自作バッファ
// MT5はCopyRatesで30秒足を取得できないため以下の2段構えで対応する：
// ① 過去分：CopyTicksRangeでTick取得 → EA内でresample(30s) → MqlRates形式で送信
// ② リアルタイム分：OnTick()でOHLCVを自前集計して蓄積
// =================================================================
struct M05Bar
{
   datetime time;    // バケット開始時刻（30秒単位に切り捨て）
   double   open;
   double   high;
   double   low;
   double   close;
   long     volume;  // ティック数で代用
};

M05Bar g_m05_bars[];          // 確定済みM0.5バーの蓄積バッファ
M05Bar g_m05_current;         // 現在形成中のM0.5バー
bool   g_m05_initialized = false;
const int G_M05_MAX_BARS = 500000; // 最大蓄積本数（約174日分）

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("========================================");
   Print("🔧 Project Forge Receiver V12.02 (M0.5: resample最適化)");
   Print("========================================");

   // 1. トレード設定
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints((int)(MaxSlippage * 10));
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   trade.SetAsyncMode(false);

   // 2. ZMQコンテキスト作成
   g_context = new Context("ProjectForge");
   if(g_context == NULL)
   {
      Print("✗ エラー: ZMQコンテキストの作成に失敗");
      return(INIT_FAILED);
   }

   // 3. 制御チャネル (REP: 5555)
   g_control_socket = new Socket(g_context, ZMQ_REP);
   if(g_control_socket == NULL || !g_control_socket.bind(ControlEndpoint))
   {
      Print("✗ エラー: 制御チャネルのバインド失敗: ", ControlEndpoint);
      return(INIT_FAILED);
   }
   Print("✓ 制御チャネル準備完了: ", ControlEndpoint);

   // 4. データチャネル (PUSH: 5556)
   g_data_socket = new Socket(g_context, ZMQ_PUSH);
   if(g_data_socket == NULL || !g_data_socket.bind(DataEndpoint))
   {
      Print("✗ エラー: データチャネルのバインド失敗: ", DataEndpoint);
      return(INIT_FAILED);
   }
   Print("✓ データチャネル準備完了: ", DataEndpoint);

   // 5. ハートビートチャネル (REP: 5558)
   g_heartbeat_socket = new Socket(g_context, ZMQ_REP);
   if(g_heartbeat_socket == NULL || !g_heartbeat_socket.bind(HeartbeatEndpoint))
   {
      Print("✗ エラー: ハートビートチャネルのバインド失敗: ", HeartbeatEndpoint);
      return(INIT_FAILED);
   }
   Print("✓ ハートビートチャネル準備完了: ", HeartbeatEndpoint);

   // 6. タイマー開始 (制御チャネルのポーリング用)
   EventSetMillisecondTimer(200);

   // 7. M0.5バッファの初期化
   ArrayResize(g_m05_bars, 0);
   g_m05_initialized = false;
   Print("✓ M0.5バッファ初期化完了");

   Print("✓ システム起動完了");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("========================================");
   Print("Project Forge Receiver V12.02 終了処理");
   Print("========================================");

   EventKillTimer();
   g_is_sending = false;

   // メモリ解放
   ArrayFree(g_history_data);
   ArrayFree(g_m05_bars);

   // ソケットとコンテキストの破棄
   if(g_data_socket != NULL)      { delete g_data_socket;      g_data_socket = NULL; }
   if(g_control_socket != NULL)   { delete g_control_socket;   g_control_socket = NULL; }
   if(g_heartbeat_socket != NULL) { delete g_heartbeat_socket; g_heartbeat_socket = NULL; }
   if(g_context != NULL)          { delete g_context;          g_context = NULL; }

   Print("✓ クリーンアップ完了");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // ハートビートのチェック (Port 5558)
   CheckHeartbeat();

   // [V12.0] M0.5バーの自作集計
   CollectM05Bar();
}

//+------------------------------------------------------------------+
//| [V12.0] M0.5(30秒足)をOnTickで自前集計する                      |
//+------------------------------------------------------------------+
void CollectM05Bar()
{
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   // 現在の30秒バケット開始時刻を計算（整数除算：ProcessHistoryRequestと統一）
   datetime bucket_time = (tick.time / 30) * 30;

   if(!g_m05_initialized)
   {
      // 初回：形成中バーを初期化
      g_m05_current.time   = bucket_time;
      g_m05_current.open   = tick.bid;
      g_m05_current.high   = tick.bid;
      g_m05_current.low    = tick.bid;
      g_m05_current.close  = tick.bid;
      g_m05_current.volume = 1;
      g_m05_initialized = true;
      return;
   }

   if(bucket_time > g_m05_current.time)
   {
      // バケット切り替わり → 前のバーを確定・蓄積
      int size = ArraySize(g_m05_bars);
      if(size >= G_M05_MAX_BARS)
      {
         // 古い先頭1万本を削除してメモリを確保
         ArrayRemove(g_m05_bars, 0, 10000);
         size = ArraySize(g_m05_bars);
      }
      ArrayResize(g_m05_bars, size + 1);
      g_m05_bars[size] = g_m05_current;

      // 新しいバーを開始
      g_m05_current.time   = bucket_time;
      g_m05_current.open   = tick.bid;
      g_m05_current.high   = tick.bid;
      g_m05_current.low    = tick.bid;
      g_m05_current.close  = tick.bid;
      g_m05_current.volume = 1;
   }
   else
   {
      // 同じバケット内：OHLCV更新
      if(tick.bid > g_m05_current.high) g_m05_current.high = tick.bid;
      if(tick.bid < g_m05_current.low)  g_m05_current.low  = tick.bid;
      g_m05_current.close = tick.bid;
      g_m05_current.volume++;
   }
}

//+------------------------------------------------------------------+
//| Expert timer function                                            |
//+------------------------------------------------------------------+
void OnTimer()
{
   // --- A. データ送信モード ---
   if(g_is_sending)
   {
      SendHistoryChunk();
      return;
   }

   // --- B. 待機モード (制御チャネルのポーリング) ---
   ZmqMsg requestMsg;
   if(!g_control_socket.recv(requestMsg, true))
   {
      return;
   }

   uchar request_bytes[];
   requestMsg.getData(request_bytes);
   string request = CharArrayToString(request_bytes);

   Print("📨 受信リクエスト(Control): ", request);

   // 1. 履歴データ転送リクエスト
   if(StringFind(request, "REQ_HISTORY") >= 0)
   {
      ProcessHistoryRequest(request);
   }
   // 2. 転送完了確認
   else if(StringFind(request, "CONFIRM_HISTORY") >= 0)
   {
      ProcessConfirmRequest();
   }
   // 3. 取引コマンド
   else if(StringFind(request, "\"type\": \"TRADE_COMMAND\"") >= 0 || StringFind(request, "\"type\":\"TRADE_COMMAND\"") >= 0)
   {
      Print("📨 取引コマンドを受信しました。処理を開始します...");
      CJAVal json;
      if(json.Deserialize(request))
      {
         CJAVal payload = json["payload"];
         ulong ticket = ExecuteTradeCommand(payload);
         if(ticket > 0)
         {
            string reply = StringFormat("{\"status\": \"ACK\", \"ticket\": %I64d}", ticket);
            SendStringResponse(g_control_socket, reply);
            Print("✓ 発注成功: Ticket=", ticket);
         }
         else
         {
            SendStringResponse(g_control_socket, "{\"status\": \"NACK\", \"reason\": \"Execution Failed\"}");
            Print("✗ 発注失敗");
         }
      }
      else
      {
         Print("✗ JSONパースエラー");
         SendStringResponse(g_control_socket, "{\"status\": \"ERROR\", \"reason\": \"Invalid JSON\"}");
      }
   }
   // 3.5. 最新M1バーリクエスト
   else if(StringFind(request, "REQUEST_M1_BAR") >= 0)
   {
      ProcessM1BarRequest();
   }
   // 4. ブローカー状態リクエスト
   else if(StringFind(request, "REQUEST_BROKER_STATE") >= 0)
   {
      ProcessBrokerStateRequest();
   }
   // 5. 直近決済履歴リクエスト
   else if(StringFind(request, "REQUEST_RECENT_HISTORY") >= 0)
   {
      ProcessRecentHistoryRequest();
   }
   // 6. その他
   else
   {
      Print("✗ 未知のコマンド: ", request);
      SendStringResponse(g_control_socket, "ERROR:Unknown Command");
   }
}

//+------------------------------------------------------------------+
//| Helper: 文字列レスポンス送信                                     |
//+------------------------------------------------------------------+
bool SendStringResponse(Socket *socket, string message)
{
   if(socket == NULL) return false;
   ZmqMsg msg(message);
   return socket.send(msg);
}

//+------------------------------------------------------------------+
//| Helper: 履歴リクエストの処理 [V12.0: M0.5対応]                  |
//+------------------------------------------------------------------+
void ProcessHistoryRequest(string request_str)
{
   int    req_bars = DATA_REQUEST_COUNT;
   string tf_name  = "M1"; // デフォルト
   string parts[];

   if(StringSplit(request_str, ':', parts) >= 3)
   {
      tf_name  = parts[1];
      req_bars = (int)StringToInteger(parts[2]);
      if(req_bars <= 0) req_bars = DATA_REQUEST_COUNT;
   }

   PrintFormat("🔄 処理開始: 履歴データ取得 (TF=%s, %d 行)...", tf_name, req_bars);

   if(ArraySize(g_history_data) > 0) ArrayFree(g_history_data);

   // =================================================================
   // [V12.1] M0.5リクエスト処理
   // ① 過去分：CopyTicksRangeでTick取得 → EA内でresample(30s) → MqlRates形式に変換
   //    Python側の受信パーサー変更不要・送信データはMqlRates形式(60bytes/bar)のまま
   // ② リアルタイム分：OnTick()で自前集計したg_m05_barsを末尾に追加
   // =================================================================
   if(tf_name == "M0.5")
   {
      int m05_realtime_count = ArraySize(g_m05_bars);

      // ① 過去分：CopyTicksRangeで必要期間のTickを取得してEA内でresample
      // req_bars本のM0.5バー = req_bars × 30秒 分のTickが必要
      // リアルタイム蓄積分で賄えない分だけTickから生成する
      int m05_history_needed = req_bars - m05_realtime_count;
      int history_generated  = 0;

      if(m05_history_needed > 0)
      {
         // 必要な期間を計算（現在時刻から何秒前まで取得するか）
         datetime time_to   = TimeCurrent();
         datetime time_from = time_to - (datetime)(m05_history_needed * 30);

         MqlTick ticks[];
         int tick_count = CopyTicksRange(_Symbol, ticks, COPY_TICKS_ALL,
                                         (ulong)time_from * 1000,  // ミリ秒単位
                                         (ulong)time_to   * 1000);

         PrintFormat("  -> CopyTicksRange: %d Tick取得 (期間: %s 〜 %s)",
                     tick_count,
                     TimeToString(time_from, TIME_DATE|TIME_SECONDS),
                     TimeToString(time_to,   TIME_DATE|TIME_SECONDS));

         if(tick_count > 0)
         {
            // Tickを30秒バケットにresample → MqlRates形式に変換
            // [最適化①] ループ前に最大サイズを事前確保してArrayResizeをループ内から排除
            // [最適化②] バケット計算を浮動小数点(MathFloor+double)から整数除算に変更
            //           3600万回のループで体感できるレベルで高速化する
            MqlRates m05_history[];
            int max_possible_bars = (int)MathCeil((double)tick_count / 2.0) + 1;
            ArrayResize(m05_history, max_possible_bars);  // 事前に最大サイズを確保

            // 整数演算でバケット計算（doubleキャスト+MathFloor不要）
            datetime cur_bucket = (ticks[0].time / 30) * 30;
            double   bkt_open   = ticks[0].bid;
            double   bkt_high   = ticks[0].bid;
            double   bkt_low    = ticks[0].bid;
            double   bkt_close  = ticks[0].bid;
            long     bkt_volume = 1;
            int      m05_count  = 0;

            for(int t = 1; t < tick_count; t++)
            {
               // [最適化②] 整数除算でバケット計算
               datetime tick_bucket = (ticks[t].time / 30) * 30;

               if(tick_bucket > cur_bucket)
               {
                  // 前のバケットを確定・保存（ArrayResizeなし）
                  m05_history[m05_count].time        = cur_bucket;
                  m05_history[m05_count].open        = bkt_open;
                  m05_history[m05_count].high        = bkt_high;
                  m05_history[m05_count].low         = bkt_low;
                  m05_history[m05_count].close       = bkt_close;
                  m05_history[m05_count].tick_volume = bkt_volume;
                  m05_history[m05_count].real_volume = 0;
                  m05_history[m05_count].spread      = 0;
                  m05_count++;

                  // 新バケット開始
                  cur_bucket = tick_bucket;
                  bkt_open   = ticks[t].bid;
                  bkt_high   = ticks[t].bid;
                  bkt_low    = ticks[t].bid;
                  bkt_close  = ticks[t].bid;
                  bkt_volume = 1;
               }
               else
               {
                  // 同バケット内：OHLCV更新
                  if(ticks[t].bid > bkt_high) bkt_high = ticks[t].bid;
                  if(ticks[t].bid < bkt_low)  bkt_low  = ticks[t].bid;
                  bkt_close = ticks[t].bid;
                  bkt_volume++;
               }
            }
            // 最後のバケットは形成中の可能性があるため追加しない（リアルタイム蓄積分と重複回避）

            // 実際のバー数にトリム
            ArrayResize(m05_history, m05_count);

            // g_history_dataにコピー
            ArrayResize(g_history_data, m05_count);
            for(int j = 0; j < m05_count; j++)
               g_history_data[j] = m05_history[j];

            history_generated = m05_count;
            PrintFormat("  -> resample完了: %d M0.5バーを生成", history_generated);
         }
         else
         {
            PrintFormat("  ⚠ CopyTicksRange失敗 (code=%d)。過去分は空で続行します。", GetLastError());
         }
      }

      // ② リアルタイム蓄積分（CollectM05Bar()で集計済み）を末尾に追加
      int existing = ArraySize(g_history_data);
      ArrayResize(g_history_data, existing + m05_realtime_count);
      for(int i = 0; i < m05_realtime_count; i++)
      {
         g_history_data[existing + i].time        = g_m05_bars[i].time;
         g_history_data[existing + i].open        = g_m05_bars[i].open;
         g_history_data[existing + i].high        = g_m05_bars[i].high;
         g_history_data[existing + i].low         = g_m05_bars[i].low;
         g_history_data[existing + i].close       = g_m05_bars[i].close;
         g_history_data[existing + i].tick_volume = g_m05_bars[i].volume;
         g_history_data[existing + i].real_volume = 0;
         g_history_data[existing + i].spread      = 0;
      }

      g_total_bars    = ArraySize(g_history_data);
      g_current_index = 0;
      int total_chunks = (int)MathCeil((double)g_total_bars / g_chunk_size);

      PrintFormat("✓ M0.5データ準備完了: %d バー (Tick由来履歴:%d本 + リアルタイム:%d本)",
                  g_total_bars, history_generated, m05_realtime_count);

      string ack_message = StringFormat(
         "ACK:TOTAL_BARS=%d;CHUNK_SIZE=%d;TOTAL_CHUNKS=%d;DATA_PORT=5556",
         g_total_bars, g_chunk_size, total_chunks
      );
      if(!SendStringResponse(g_control_socket, ack_message))
      {
         Print("✗ エラー: ACK送信失敗");
         return;
      }
   }
   // =================================================================
   // 通常のM1リクエスト処理（従来通り）
   // =================================================================
   else
   {
      int copied = CopyRates(_Symbol, PERIOD_M1, 1, req_bars, g_history_data);
      if(copied <= 0)
      {
         int err = GetLastError();
         string err_msg = StringFormat("NACK:CopyRates failed code=%d", err);
         Print(err_msg);
         SendStringResponse(g_control_socket, err_msg);
         return;
      }

      g_total_bars    = copied;
      g_current_index = 0;
      int total_chunks = (int)MathCeil((double)g_total_bars / g_chunk_size);

      PrintFormat("✓ データ取得成功: %d バー, チャンク数: %d", g_total_bars, total_chunks);

      string ack_message = StringFormat(
         "ACK:TOTAL_BARS=%d;CHUNK_SIZE=%d;TOTAL_CHUNKS=%d;DATA_PORT=5556",
         g_total_bars, g_chunk_size, total_chunks
      );
      if(!SendStringResponse(g_control_socket, ack_message))
      {
         Print("✗ エラー: ACK送信失敗");
         return;
      }
   }

   // データ送信ステートマシンを起動
   g_is_sending = true;
   EventSetMillisecondTimer(10);
   Print("🚀 データ転送を開始します (PUSH: 5556)...");
}

//+------------------------------------------------------------------+
//| Helper: 転送完了確認の処理                                       |
//+------------------------------------------------------------------+
void ProcessConfirmRequest()
{
   Print("✅ 完了通知を受信: メモリを解放します。");
   ArrayFree(g_history_data);
   g_total_bars    = 0;
   g_current_index = 0;
   SendStringResponse(g_control_socket, "ACK_CONFIRMED");
}

//+------------------------------------------------------------------+
//| Helper: データチャンク送信 (ゼロ・シリアライズ実装)              |
//+------------------------------------------------------------------+
void SendHistoryChunk()
{
   if(g_current_index >= g_total_bars)
   {
      Print("🏁 全チャンクの送信完了。EOSシグナルを送信します。");
      g_is_sending = false;
      EventSetMillisecondTimer(200);
      SendStringResponse(g_data_socket, "END_OF_STREAM");
      return;
   }

   int bars_to_send  = MathMin(g_chunk_size, g_total_bars - g_current_index);
   int start_offset  = g_current_index;

   uchar byte_chunk[];
   int chunk_byte_size = bars_to_send * MQLRATES_STRUCT_SIZE;

   ArrayFree(byte_chunk);
   if(ArrayResize(byte_chunk, chunk_byte_size) != chunk_byte_size)
   {
      Print("✗ エラー: チャンク用メモリ確保に失敗");
      g_is_sending = false;
      EventSetMillisecondTimer(1000);
      return;
   }

   for(int i = 0; i < bars_to_send; i++)
   {
      if(!StructToCharArray(g_history_data[start_offset + i], byte_chunk, i * MQLRATES_STRUCT_SIZE))
      {
         Print("✗ エラー: StructToCharArray 失敗 at index ", i);
      }
   }

   if(ArraySize(byte_chunk) == 0)
   {
      Print("⚠ 警告: 送信しようとしたバイト配列が空です！");
      return;
   }

   ZmqMsg chunkMsg(chunk_byte_size);
   chunkMsg.setData(byte_chunk);

   if(!g_data_socket.send(chunkMsg))
   {
      Print("⚠ 警告: チャンク送信失敗 (次回再試行)");
      return;
   }

   if(g_current_index == 0)
   {
      PrintFormat("DEBUG: 初回チャンク送信 - Bars: %d, Bytes: %d", bars_to_send, ArraySize(byte_chunk));
   }

   g_current_index += bars_to_send;
}

//+------------------------------------------------------------------+
//| Helper: ハートビート処理 (Port 5558)                             |
//+------------------------------------------------------------------+
void CheckHeartbeat()
{
   if(g_heartbeat_socket == NULL) return;

   ZmqMsg requestMsg;
   if(g_heartbeat_socket.recv(requestMsg, true))
   {
      uchar request_bytes[];
      requestMsg.getData(request_bytes);
      string msg = CharArrayToString(request_bytes);

      if(StringFind(msg, "PING") >= 0)
      {
         string pong = "PONG:" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);
         SendStringResponse(g_heartbeat_socket, pong);
      }
      else
      {
         SendStringResponse(g_heartbeat_socket, "NACK");
      }
   }
}

//+------------------------------------------------------------------+
//| Helper: ブローカー状態をJSON形式で返信                           |
//+------------------------------------------------------------------+
void ProcessBrokerStateRequest()
{
   double equity      = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance     = AccountInfoDouble(ACCOUNT_BALANCE);
   double margin      = AccountInfoDouble(ACCOUNT_MARGIN);
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);

   string positions_json = "[";
   int total = PositionsTotal();
   int count = 0;

   for(int i=0; i<total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(count > 0) positions_json += ",";
         string symbol = PositionGetString(POSITION_SYMBOL);
         long type     = PositionGetInteger(POSITION_TYPE);
         string dir    = (type == POSITION_TYPE_BUY) ? "BUY" : "SELL";
         double lots   = PositionGetDouble(POSITION_VOLUME);
         double price  = PositionGetDouble(POSITION_PRICE_OPEN);
         double sl     = PositionGetDouble(POSITION_SL);
         double tp     = PositionGetDouble(POSITION_TP);
         double profit = PositionGetDouble(POSITION_PROFIT);
         long time     = PositionGetInteger(POSITION_TIME);
         string time_str = TimeToString(time, TIME_DATE|TIME_SECONDS);

         string p = StringFormat(
            "{\"ticket\":%I64u,\"symbol\":\"%s\",\"direction\":\"%s\",\"lots\":%.2f,\"entry_price\":%.5f,\"stop_loss\":%.5f,\"take_profit\":%.5f,\"unrealized_pnl\":%.2f,\"entry_time\":\"%s\"}",
            ticket, symbol, dir, lots, price, sl, tp, profit, time_str
         );
         positions_json += p;
         count++;
      }
   }
   positions_json += "]";

   string response = StringFormat(
      "{\"equity\":%.2f,\"balance\":%.2f,\"margin\":%.2f,\"free_margin\":%.2f,\"positions\":%s}",
      equity, balance, margin, free_margin, positions_json
   );

   SendStringResponse(g_control_socket, response);
   Print("✓ 状態同期データを送信しました: Equity=", equity, " Positions=", total);
}

//+------------------------------------------------------------------+
//| Helper: 最新M1バーをJSON形式で返信 (リアルタイム監視用)          |
//+------------------------------------------------------------------+
void ProcessM1BarRequest()
{
   MqlRates rates[];
   // 完全に確定した直近のバー(Shift 1)を1本取得
   int copied = CopyRates(_Symbol, PERIOD_M1, 1, 1, rates);

   if(copied > 0)
   {
      // 直近5ティックの出来高平均を計算
      MqlTick ticks[];
      double tick_vol_mean_5 = 0.0;
      int copied_ticks = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, 0, 5);
      if(copied_ticks > 0)
      {
         double sum_vol = 0;
         for(int i = 0; i < copied_ticks; i++)
         {
            sum_vol += (double)ticks[i].volume;
         }
         tick_vol_mean_5 = sum_vol / copied_ticks;
      }

      // リアルタイムスプレッドの取得 (ポイントからpipsへ変換)
      long spread_points = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
      double current_spread_pips = (double)spread_points / 10.0;

      string response = StringFormat(
         "{\"time\":%I64d,\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"tick_volume\":%I64d,\"real_volume\":%I64d,\"tick_volume_mean_5\":%.2f,\"spread\":%.1f}",
         rates[0].time,
         rates[0].open,
         rates[0].high,
         rates[0].low,
         rates[0].close,
         rates[0].tick_volume,
         rates[0].real_volume,
         tick_vol_mean_5,
         current_spread_pips
      );
      SendStringResponse(g_control_socket, response);
   }
   else
   {
      Print("✗ M1バー取得失敗");
      SendStringResponse(g_control_socket, "ERROR:CopyRates Failed");
   }
}

//+------------------------------------------------------------------+
//| Helper: 取引コマンドの実行                                       |
//+------------------------------------------------------------------+
ulong ExecuteTradeCommand(CJAVal &payload)
{
   string action = payload["action"].ToStr();
   double lots   = payload["lots"].ToDbl();
   double sl     = payload["stop_loss"].ToDbl();
   double tp     = payload["take_profit"].ToDbl();

   PrintFormat("▶ 注文実行: %s %.2f Lots, SL=%.3f, TP=%.3f", action, lots, sl, tp);

   if(action == "BUY")
   {
      if(trade.Buy(lots, _Symbol, 0, sl, tp, "ProjectForge V12"))
         return trade.ResultOrder();
   }
   else if(action == "SELL")
   {
      if(trade.Sell(lots, _Symbol, 0, sl, tp, "ProjectForge V12"))
         return trade.ResultOrder();
   }
   else if(action == "HOLD")
   {
      return 1;
   }
   else if(action == "CLOSE")
   {
      ulong target_ticket = (ulong)payload["ticket"].ToInt();
      if(target_ticket > 0)
      {
         if(trade.PositionClose(target_ticket))
         {
            Print("✓ タイムアウト決済完了: Ticket=", target_ticket);
            return trade.ResultDeal();
         }
      }
   }

   Print("✗ 注文エラー: ", GetLastError());
   return 0;
}

//+------------------------------------------------------------------+
//| Helper: 直近決済履歴をJSON形式で返信 (サイレントクローズ対策)    |
//+------------------------------------------------------------------+
void ProcessRecentHistoryRequest()
{
   string history_json = "[";
   datetime end_time   = TimeCurrent();
   datetime start_time = end_time - 3600; // 過去1時間分

   HistorySelect(start_time, end_time);
   int total = HistoryDealsTotal();
   int count = 0;

   for(int i = 0; i < total; i++)
   {
      ulong deal_ticket = HistoryDealGetTicket(i);
      long entry_type   = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);

      if(entry_type == DEAL_ENTRY_OUT || entry_type == DEAL_ENTRY_OUT_BY)
      {
         ulong position_ticket = HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID);
         long reason           = HistoryDealGetInteger(deal_ticket, DEAL_REASON);

         string close_reason = "UNKNOWN";
         if(reason == DEAL_REASON_SL)     close_reason = "SL";
         else if(reason == DEAL_REASON_TP)     close_reason = "PT";
         else if(reason == DEAL_REASON_EXPERT) close_reason = "TO";

         if(count > 0) history_json += ",";
         history_json += StringFormat("{\"ticket\":%I64u,\"close_reason\":\"%s\"}", position_ticket, close_reason);
         count++;
      }
   }
   history_json += "]";
   SendStringResponse(g_control_socket, history_json);
}
