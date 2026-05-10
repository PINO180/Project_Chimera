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
Socket  *g_m3_notify_socket = NULL; // PUSH: 5557 M3確定通知専用

// --- M3確定検出用 ---
datetime g_last_m3_bucket     = 0;     // 最後に通知したM3バケット時刻
bool     g_python_ready        = false; // Python側の準備完了フラグ
// [STALE-GUARD] CONFIRM_HISTORY を受信するまで M3 通知を送信しない。
// Python 側のウォームアップ（最大30分）中にキューが溜まり、
// 完了直後に大量のオーダーが一斉発射される事故を EA 側から根本防止する。

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

   // 6. M3通知チャネル (PUSH: 5557)
   g_m3_notify_socket = new Socket(g_context, ZMQ_PUSH);
   if(g_m3_notify_socket == NULL || !g_m3_notify_socket.bind("tcp://*:5557"))
   {
      Print("✗ エラー: M3通知チャネルのバインド失敗: tcp://*:5557");
      return(INIT_FAILED);
   }
   Print("✓ M3通知チャネル準備完了: tcp://*:5557");

   // 7. タイマー開始
   // [LAG-FIX] 周期 200ms → 50ms に短縮。
   //   旧: M3境界検出の最大遅延が 200ms
   //   新: M3境界検出の最大遅延が 50ms
   //   制御チャネル (ControlPoll) の応答性も向上 (ハンドシェイク待ちが減る)
   EventSetMillisecondTimer(50);

   // 8. M0.5バッファの初期化
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
   if(g_m3_notify_socket != NULL) { delete g_m3_notify_socket; g_m3_notify_socket = NULL; }
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

   // 中値を使用（学習側の mid_price = (bid + ask) / 2 と一致させる）
   double mid = (tick.bid + tick.ask) / 2.0;

   // 現在の30秒バケット開始時刻を計算（整数除算：ProcessHistoryRequestと統一）
   datetime bucket_time = (tick.time / 30) * 30;

   if(!g_m05_initialized)
   {
      // 初回：形成中バーを初期化
      // [VOLUME-FIX] ECN ブローカーは tick.volume = real_volume = 0 のため、
      // 学習側 (s1_1_B_build_ohlcv.py) と同じく tick 数を volume として使用する。
      // 学習側は CSV の VOLUME 空 → 0 → s1_1_B で tick_count に fallback 修正済み。
      // 本番側もここで「tick 1個 = volume 1」として揃える。
      g_m05_current.time   = bucket_time;
      g_m05_current.open   = mid;
      g_m05_current.high   = mid;
      g_m05_current.low    = mid;
      g_m05_current.close  = mid;
      g_m05_current.volume = 1;  // [VOLUME-FIX] tick 数カウント (旧: (long)tick.volume)
      g_m05_initialized = true;
      return;
   }

   if(bucket_time > g_m05_current.time)
   {
      // バケット切り替わり → 前のバーを確定・蓄積
      // [LAG-FIX-3] OnTimer 側 (L296-298) の volume > 0 ガードと整合させる。
      //   Phase 9 #54 で OnTimer が g_m05_current を「volume=0 スタブ」として
      //   引き継ぐ設計を導入した結果、silent → tick 復帰の境界 (この CollectM05Bar
      //   の new-bucket 分岐) で V=0 stub をそのまま push してしまう経路が存在した。
      //   学習側 s1_1_B_build_ohlcv.py の `filter(tick_count > 0)` と完全整合させるため、
      //   volume == 0 のバーは g_m05_bars に追加せず破棄する。
      if(g_m05_current.volume > 0)
      {
         int size = ArraySize(g_m05_bars);
         if(size >= G_M05_MAX_BARS)
         {
            // 古い先頭1万本を削除してメモリを確保
            ArrayRemove(g_m05_bars, 0, 10000);
            size = ArraySize(g_m05_bars);
         }
         ArrayResize(g_m05_bars, size + 1);
         g_m05_bars[size] = g_m05_current;
      }
      // ※ volume == 0 のとき (= OnTimer で作られた V=0 stub のまま tick が来なかった
      //    bucket) は何もせず破棄。silent 期間の bucket は g_m05_bars に存在しない
      //    状態 = 学習側 filter(tick_count > 0) と物理的に同じ挙動になる。

      // 新しいバーを開始
      // [VOLUME-FIX] ECN ブローカーは tick.volume = 0 のため tick 数で代用 (学習側と統一)
      g_m05_current.time   = bucket_time;
      g_m05_current.open   = mid;
      g_m05_current.high   = mid;
      g_m05_current.low    = mid;
      g_m05_current.close  = mid;
      g_m05_current.volume = 1;  // [VOLUME-FIX] tick 1個目 (旧: (long)tick.volume)
   }
   else
   {
      // 同じバケット内：OHLCV更新
      // [VOLUME-FIX] ECN ブローカーは tick.volume = 0 のため tick 数を加算 (学習側と統一)
      // [LAG-FIX-2] OnTimer の強制確定で volume=0 のスタブが作られた場合は、
      //   最初のティック値で OHLCV を上書き (forward fill された prev_close を破棄)。
      if(g_m05_current.volume == 0)
      {
         // OnTimer で初期化されたスタブバー: 最初のティック値で上書き
         g_m05_current.open   = mid;
         g_m05_current.high   = mid;
         g_m05_current.low    = mid;
         g_m05_current.close  = mid;
         g_m05_current.volume = 1;
      }
      else
      {
         if(mid > g_m05_current.high) g_m05_current.high = mid;
         if(mid < g_m05_current.low)  g_m05_current.low  = mid;
         g_m05_current.close = mid;
         g_m05_current.volume += 1;  // [VOLUME-FIX] tick 数カウント (旧: (long)tick.volume)
      }
   }
}

//+------------------------------------------------------------------+
//| Expert timer function                                            |
//+------------------------------------------------------------------+
void OnTimer()
{
   // --- M3確定チェック（bucket_timeベース：即時検出）---
   // [LAG-FIX] TimeCurrent() → TimeTradeServer() に変更
   //   旧: TimeCurrent() = 直近ティックの時刻
   //       → ティックが秒単位で間欠的に来る (XAU/USD 深夜帯) と
   //          M3境界を跨いでも値が更新されず、検出が大幅に遅延
   //          (実機で 2062ms 遅延、age=129秒 STALE-GUARD 発火を確認)
   //   新: TimeTradeServer() = 取引サーバー時刻 (ティック非依存で進む)
   //       → 50msタイマーが毎回チェック → 最大遅延 50ms
   if(g_m3_notify_socket != NULL && g_python_ready)
   {
      datetime now = TimeTradeServer();
      datetime current_m05_bucket = (now / 30) * 30;
      datetime current_m3 = (now / 180) * 180;

      // [LAG-FIX-2] M0.5 境界を跨いだら g_m05_current を強制確定
      //
      // 旧: ティック到来を待って CollectM05Bar が bucket 切替を検知 → M0.5 バーを g_m05_bars に追加
      //     → ティック間隔が秒単位だと、M3 close から最初のティックまで数百ms〜1秒以上遅延
      //     (実機ログで M3 close → process_new_m05_bar 開始まで 631ms 確認)
      //
      // 新: TimeTradeServer ベースで M0.5 境界 (30秒) 経過を検知し、
      //     g_m05_current.volume > 0 なら強制確定して g_m05_bars に追加。
      //     新バケットは前 close を引き継ぎ、volume=0 のスタブとして残す
      //     (次のティックで CollectM05Bar が OHLCV を上書きする)。
      //
      // 学習側整合性: s1_1_B_build_ohlcv.py L224 で `filter(tick_count > 0)` により
      //              volume=0 バーは破棄される。本実装では volume>0 のバーのみ確定するため
      //              学習側と完全整合 (volume=0 スタブは送信しない)。
      if(g_m05_initialized
         && current_m05_bucket > g_m05_current.time
         && g_m05_current.volume > 0)
      {
         int size = ArraySize(g_m05_bars);
         if(size >= G_M05_MAX_BARS)
         {
            ArrayRemove(g_m05_bars, 0, 10000);
            size = ArraySize(g_m05_bars);
         }
         ArrayResize(g_m05_bars, size + 1);
         g_m05_bars[size] = g_m05_current;

         // 新バケットを volume=0 スタブとして開始 (前バーの close を引き継ぎ)
         double prev_close = g_m05_current.close;
         g_m05_current.time   = current_m05_bucket;
         g_m05_current.open   = prev_close;
         g_m05_current.high   = prev_close;
         g_m05_current.low    = prev_close;
         g_m05_current.close  = prev_close;
         g_m05_current.volume = 0;  // 次のティックで CollectM05Bar が上書きする
      }

      if(current_m3 > g_last_m3_bucket)
      {
         g_last_m3_bucket = current_m3;
         int buf_size = ArraySize(g_m05_bars);
         if(buf_size > 0)
         {
            M05Bar last_bar = g_m05_bars[buf_size - 1];
            long spread_points = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
            double notify_spread_pips = (double)spread_points / 10.0;
            string notify = StringFormat(
               "{\"time\":%I64d,\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%I64d,\"spread\":%.1f}",
               last_bar.time,
               last_bar.open,
               last_bar.high,
               last_bar.low,
               last_bar.close,
               last_bar.volume,
               notify_spread_pips
            );
            ZmqMsg notifyMsg(notify);
            g_m3_notify_socket.send(notifyMsg);
            PrintFormat("📡 M3確定通知送信(Timer): M3_bucket=%s, M0.5=%s",
                        TimeToString(current_m3, TIME_DATE|TIME_SECONDS),
                        TimeToString(last_bar.time, TIME_DATE|TIME_SECONDS));
         }
      }
   }

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
            long ack_spread_points = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
            double ack_spread_pips = (double)ack_spread_points / 10.0;
            string reply = StringFormat("{\"status\": \"ACK\", \"ticket\": %I64d, \"spread\": %.1f}", ticket, ack_spread_pips);
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
            // [TRAIN-SERVE-FIX] 学習側(s1_1_B_build_ohlcv.py)はmid_price=(bid+ask)/2でM0.5バーを生成。
            // ここでもbidではなくmid_priceを使用することで学習/本番のM0.5バー価格を完全一致させる。
            // [VOLUME-FIX] ECN ブローカーは ticks[].volume = 0 のため tick 数で代用 (学習側と統一)。
            datetime cur_bucket = (ticks[0].time / 30) * 30;
            double   bkt_mid_0  = (ticks[0].bid + ticks[0].ask) / 2.0;
            double   bkt_open   = bkt_mid_0;
            double   bkt_high   = bkt_mid_0;
            double   bkt_low    = bkt_mid_0;
            double   bkt_close  = bkt_mid_0;
            long     bkt_volume = 1;  // [VOLUME-FIX] tick 1個目 (旧: (long)ticks[0].volume)
            int      m05_count  = 0;

            for(int t = 1; t < tick_count; t++)
            {
               // [最適化②] 整数除算でバケット計算
               datetime tick_bucket = (ticks[t].time / 30) * 30;
               // [TRAIN-SERVE-FIX] mid_priceを計算
               double tick_mid = (ticks[t].bid + ticks[t].ask) / 2.0;

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
                  // [VOLUME-FIX] ECN ブローカーは ticks[].volume = 0 のため tick 数で代用 (学習側と統一)
                  cur_bucket = tick_bucket;
                  bkt_open   = tick_mid;
                  bkt_high   = tick_mid;
                  bkt_low    = tick_mid;
                  bkt_close  = tick_mid;
                  bkt_volume = 1;  // [VOLUME-FIX] tick 1個目 (旧: (long)ticks[t].volume)
               }
               else
               {
                  // 同バケット内：OHLCV更新
                  // [VOLUME-FIX] ECN ブローカーは ticks[].volume = 0 のため tick 数を加算 (学習側と統一)
                  if(tick_mid > bkt_high) bkt_high = tick_mid;
                  if(tick_mid < bkt_low)  bkt_low  = tick_mid;
                  bkt_close = tick_mid;
                  bkt_volume += 1;  // [VOLUME-FIX] tick 数カウント (旧: (long)ticks[t].volume)
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
   // [STALE-GUARD] ここでは g_python_ready を true にしない。
   // データ転送完了 ≠ Python ウォームアップ完了。
   // M3通知の解禁は Python 側から NOTIFY_PYTHON_READY を受信したタイミングのみで行う。
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
         // [STALE-GUARD] PING:READY / PING:NOT_READY でg_python_readyを毎回同期する。
         // EA再起動・瞬断後も次のHeartbeat受信時に自動的に正しい状態に収束する。
         bool prev_ready = g_python_ready;
         if(StringFind(msg, "PING:READY") >= 0)
         {
            g_python_ready = true;
            if(!prev_ready)
               Print("🟢 [STALE-GUARD] Heartbeat経由でPython準備完了を確認。M3通知を解禁します。");
         }
         else if(StringFind(msg, "PING:NOT_READY") >= 0)
         {
            g_python_ready = false;
            if(prev_ready)
               Print("🔴 [STALE-GUARD] Heartbeat経由でPython未準備を確認。M3通知をブロックします。");
         }
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

   // [Phase4: 絶対価格バリアへの回帰]
   // 旧 SL/TP-FIX は STALE-GUARD 未実装時代の応急処置だった。
   // STALE-GUARD で M3 close と約定価格のズレが ±$0.5 以内に収まった今、
   // Python が M3 close 基準で計算した『絶対価格バリア』をそのまま使うべき。
   //
   // 理由:
   //   ラベリング (create_proxy_labels) は M3 close 基準で PT/SL を判定する。
   //   シミュレーター (backtest_simulator L887-919) も M3 close 基準で PnL を計算する。
   //   つまり AI は『M3 close 起点の絶対バリア』に到達する確率を学習している。
   //   約定価格基準 (sl_width/tp_width) で再計算すると、TP の絶対位置が学習想定からズレる。
   //
   // 実測: TO決済10件中7件 (70%) で TP がシミュレーター想定より遠ざかっていた
   //       (avg 148pips、max 447pips)。これが TO率41% の主因。
   //
   // 注: sl_width/tp_width はデバッグログ用に Python から送信され続けるが、
   //     OrderSend では使用しない (絶対価格 stop_loss/take_profit のみを尊重)。
   double final_sl   = payload["stop_loss"].ToDbl();
   double final_tp   = payload["take_profit"].ToDbl();
   // デバッグ参考用: Python が M3 close 基準で計算した期待バリア幅
   double py_sl_width = payload["sl_width"].ToDbl();
   double py_tp_width = payload["tp_width"].ToDbl();

   if(action == "BUY")
   {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      // 約定後の実バリア幅 (デバッグ参考用、ペイオフ比のスリッページ揺れを観測する)
      double actual_sl_width = ask - final_sl;
      double actual_tp_width = final_tp - ask;
      // [TP-SANITY-CHECK] BUY における TP は約定価格 (Ask) より上にあるべき。
      //   処理ラグ間に市場が予想と逆方向 (= 上昇) に大きく動いた場合、
      //   M3 close 基準で計算した TP が約定 Ask より下に来ることがある。
      //   このまま OrderSend すると即時に逆方向 TP として決済され、
      //   ブローカーは「TP 決済 = 利確」と記録するが実際には損失となる
      //   (Excel の `決済指値(T/P)` がエントリーより下に置かれる現象)。
      //   学習側にはそもそも処理ラグが存在しないため、TP距離 ≤ 0 は学習で発生不能。
      //   よって本番でこの条件を弾いても Train-Serve Skew は生まない。
      if(actual_tp_width <= 0.0)
      {
         PrintFormat("⚠ [TP-SANITY-CHECK] BUY 注文を破棄: TP(%.3f) が Ask(%.3f) 以下 (実TP幅=%.3f)。処理ラグ間の市場逆行が原因。",
                     final_tp, ask, actual_tp_width);
         SendStringResponse(g_control_socket, "{\"status\": \"NACK\", \"reason\": \"TP_REVERSED_BY_LAG\"}");
         return 0;
      }
      PrintFormat("▶ 注文実行: BUY %.2f Lots, Ask=%.3f, SL=%.3f(実幅%.3f/Py幅%.3f), TP=%.3f(実幅%.3f/Py幅%.3f)",
                  lots, ask, final_sl, actual_sl_width, py_sl_width,
                  final_tp, actual_tp_width, py_tp_width);
      if(trade.Buy(lots, _Symbol, 0, final_sl, final_tp, "ProjectForge V12"))
         return trade.ResultOrder();

      // [DIAG] trade.Buy 失敗時の retcode を Python 側に伝搬する
      //   旧: NACK の reason は単に "Execution Failed" のみ → 真因 (broker 側の retcode) が
      //       Python ログから判別不能。例えば 5/8 22:00〜 のアルゴリズム取引 OFF 事象は
      //       retcode=10027 (CLIENT_DISABLES_AT) を出していたはずだが、Python ログには
      //       "Execution Failed" としか残らず、原因切り分けに 3 時間以上要した。
      //   新: NACK reason に retcode を含める。代表的な retcode:
      //       10004 REQUOTE / 10006 REJECT / 10013 INVALID / 10014 INVALID_VOLUME /
      //       10015 INVALID_PRICE / 10016 INVALID_STOPS / 10017 TRADE_DISABLED /
      //       10018 MARKET_CLOSED / 10019 NO_MONEY / 10020 PRICE_CHANGED /
      //       10021 PRICE_OFF / 10024 TOO_MANY_REQUESTS /
      //       10026 SERVER_DISABLES_AT / 10027 CLIENT_DISABLES_AT (= ターミナル取引許可OFF)
      uint buy_retcode = trade.ResultRetcode();
      string buy_comment = trade.ResultComment();
      PrintFormat("✗ trade.Buy 失敗: retcode=%u, comment='%s', last_error=%d",
                  buy_retcode, buy_comment, GetLastError());
      string buy_reply = StringFormat(
         "{\"status\": \"NACK\", \"reason\": \"TRADE_FAILED_%u\", \"retcode\": %u, \"comment\": \"%s\"}",
         buy_retcode, buy_retcode, buy_comment);
      SendStringResponse(g_control_socket, buy_reply);
      return 0;
   }
   else if(action == "SELL")
   {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      // 約定後の実バリア幅 (デバッグ参考用、ペイオフ比のスリッページ揺れを観測する)
      double actual_sl_width = final_sl - bid;
      double actual_tp_width = bid - final_tp;
      // [TP-SANITY-CHECK] SELL における TP は約定価格 (Bid) より下にあるべき。
      //   処理ラグ間に市場が予想と逆方向 (= 下降) に大きく動いた場合、
      //   M3 close 基準で計算した TP が約定 Bid より上に来ることがある。
      //   このまま OrderSend すると即時に逆方向 TP として決済され、
      //   ブローカーは「TP 決済 = 利確」と記録するが実際には損失となる。
      //   学習側にはそもそも処理ラグが存在しないため、TP距離 ≤ 0 は学習で発生不能。
      //   よって本番でこの条件を弾いても Train-Serve Skew は生まない。
      if(actual_tp_width <= 0.0)
      {
         PrintFormat("⚠ [TP-SANITY-CHECK] SELL 注文を破棄: TP(%.3f) が Bid(%.3f) 以上 (実TP幅=%.3f)。処理ラグ間の市場逆行が原因。",
                     final_tp, bid, actual_tp_width);
         SendStringResponse(g_control_socket, "{\"status\": \"NACK\", \"reason\": \"TP_REVERSED_BY_LAG\"}");
         return 0;
      }
      PrintFormat("▶ 注文実行: SELL %.2f Lots, Bid=%.3f, SL=%.3f(実幅%.3f/Py幅%.3f), TP=%.3f(実幅%.3f/Py幅%.3f)",
                  lots, bid, final_sl, actual_sl_width, py_sl_width,
                  final_tp, actual_tp_width, py_tp_width);
      if(trade.Sell(lots, _Symbol, 0, final_sl, final_tp, "ProjectForge V12"))
         return trade.ResultOrder();

      // [DIAG] trade.Sell 失敗時の retcode を Python 側に伝搬する (BUY 側と同等)
      uint sell_retcode = trade.ResultRetcode();
      string sell_comment = trade.ResultComment();
      PrintFormat("✗ trade.Sell 失敗: retcode=%u, comment='%s', last_error=%d",
                  sell_retcode, sell_comment, GetLastError());
      string sell_reply = StringFormat(
         "{\"status\": \"NACK\", \"reason\": \"TRADE_FAILED_%u\", \"retcode\": %u, \"comment\": \"%s\"}",
         sell_retcode, sell_retcode, sell_comment);
      SendStringResponse(g_control_socket, sell_reply);
      return 0;
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
