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
// [TICK-AGG-FIX A-2] live の M0.5 を OnTick 自前集計から CopyTicksRange ベースに
// 統一するための追跡変数。直近で「確定済み (= g_m05_bars に push 済み)」 の
// M0.5 バケット開始時刻。これより新しい完成バケットを OnTimer で順次確定する。
// 初期値 0 = 未確定 (最初の確定で warmup 末尾 or 現在時刻基準にセット)。
datetime g_last_finalized_m05_bucket = 0;
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
// [TICK-AGG-FIX A-2] g_m05_current (旧 OnTick 形成中バー) は廃止。
// OnTimer が CopyTicksRange で完成バケットを直接構築するため不要。
// 宣言のみ残置 (他参照なし)。将来削除可。
M05Bar g_m05_current;
bool   g_m05_initialized = false;
const int G_M05_MAX_BARS = 500000; // 最大蓄積本数（約174日分）

//+------------------------------------------------------------------+
//| [TICK-AGG-FIX A-2] 単一バケットを CopyTicksRange で再構築する     |
//|                                                                  |
//| 指定した 30 秒バケット [bucket_start, bucket_start+30) の全 tick を |
//| CopyTicksRange (絶対時間範囲) で取得し、学習側 s1_1_B / warmup    |
//| ProcessHistoryRequest と完全に同一のロジックで OHLCV を構築する。  |
//|                                                                  |
//| SSoT 集約ルール (学習側 s1_1_B L172 / warmup L552-598 と bit 一致):|
//|   - mid = (bid + ask) / 2                                        |
//|   - open=最初の mid, high=max, low=min, close=最後の mid         |
//|   - volume = tick 数 (ECN は tick.volume=0 のため tick 数で代用)  |
//|                                                                  |
//| 前回失敗 (ExecuteTickRecovery) との違い: count-based offset では  |
//| なく絶対時間範囲 [start_ms, end_ms) で取得するため、バックグラウンド |
//| tick 補完による配列インデックスのシフトに影響されない。           |
//|                                                                  |
//| 戻り値: true=バケット内に tick あり (out_bar 有効) / false=tick 0  |
//|         (学習側 filter(tick_count>0) と整合し、空バケットは破棄)   |
//+------------------------------------------------------------------+
bool BuildM05BarFromTicks(datetime bucket_start, M05Bar &out_bar)
{
   ulong from_ms = (ulong)bucket_start * 1000;
   ulong to_ms   = (ulong)(bucket_start + 30) * 1000;  // [start, start+30秒)

   MqlTick ticks[];
   // CopyTicksRange の to は exclusive ではないため、30秒バケットの終端は
   // 次バケット開始の 1ms 前 (to_ms - 1) までを指定して厳密に [start, start+30) とする。
   int tick_count = CopyTicksRange(_Symbol, ticks, COPY_TICKS_ALL,
                                   from_ms, to_ms - 1);

   if(tick_count <= 0)
      return false;  // tick なし = 空バケット (学習側 filter(tick_count>0) と整合)

   double mid0 = (ticks[0].bid + ticks[0].ask) / 2.0;
   double bkt_open  = mid0;
   double bkt_high  = mid0;
   double bkt_low   = mid0;
   double bkt_close = mid0;
   long   bkt_vol   = 1;

   for(int t = 1; t < tick_count; t++)
   {
      double mid = (ticks[t].bid + ticks[t].ask) / 2.0;
      if(mid > bkt_high) bkt_high = mid;
      if(mid < bkt_low)  bkt_low  = mid;
      bkt_close = mid;
      bkt_vol  += 1;
   }

   out_bar.time   = bucket_start;
   out_bar.open   = bkt_open;
   out_bar.high   = bkt_high;
   out_bar.low    = bkt_low;
   out_bar.close  = bkt_close;
   out_bar.volume = bkt_vol;
   return true;
}

//+------------------------------------------------------------------+
//| [TICK-AGG-FIX A-2] g_m05_bars 末尾へ 1 本追加 (上限管理込み)      |
//+------------------------------------------------------------------+
void PushM05Bar(const M05Bar &bar)
{
   int size = ArraySize(g_m05_bars);
   if(size >= G_M05_MAX_BARS)
   {
      ArrayRemove(g_m05_bars, 0, 10000);
      size = ArraySize(g_m05_bars);
   }
   ArrayResize(g_m05_bars, size + 1);
   g_m05_bars[size] = bar;
}

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
   // [TICK-AGG-FIX A-2] 確定基準点もリセット。EA 再起動時に前回値が残ると
   // OnTimer 初回判定 (==0) が狂い基準点設定がスキップされるため明示的に 0 に戻す。
   g_last_finalized_m05_bucket = 0;
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

   // [TICK-AGG-FIX A-2] M0.5 バーの集計は OnTick 自前積み上げ (旧 CollectM05Bar)
   // を廃止し、OnTimer で完成バケットを CopyTicksRange から再構築する方式へ移行。
   // 理由: OnTick coalescing で中間 tick が EA に届かず、自前積み上げの close が
   //   学習側 (= CopyTicksRange で全 tick 集約) と 39% しか一致しなかった
   //   (warmup は 99.5% 一致)。OnTick 集約そのものを廃止し coalescing 問題を根絶。
   // M0.5 確定は OnTimer (TimeTradeServer ベース) が担う。
}

//+------------------------------------------------------------------+
//| [TICK-AGG-FIX A-2] CollectM05Bar (OnTick 自前集計) は廃止          |
//|                                                                  |
//| 旧実装は OnTick ごとに tick の mid を g_m05_current に積み上げて   |
//| M0.5 バーを作っていたが、MT5 の OnTick coalescing で中間 tick が   |
//| EA に届かず、close が学習側 (CopyTicksRange 全 tick 集約) と       |
//| 39% しか一致しなかった (warmup は 99.5% 一致)。                    |
//|                                                                  |
//| A-2 では OnTimer が完成バケットを CopyTicksRange で再構築する      |
//| (BuildM05BarFromTicks)。OnTick 集約を完全に廃止することで          |
//| coalescing 問題を根絶し、live の M0.5 を warmup/学習側と bit       |
//| 一致させる。g_m05_current も本関数とともに不要となった。          |
//| (復活させると非対称が再発するため、関数ごと削除して呼び戻しを防ぐ) |
//+------------------------------------------------------------------+

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

      // [TICK-AGG-FIX A-2] 完成バケットを CopyTicksRange で再構築して確定
      //
      // 旧 (LAG-FIX-2): OnTick 自前積み上げの g_m05_current を境界越えで確定。
      //   → OnTick coalescing で中間 tick を取りこぼし、close が学習側と 39% しか
      //     一致しない (warmup は CopyTicksRange で 99.5% 一致)。
      //
      // 新 (A-2): 「形成中バケット (current_m05_bucket) は確定しない」。
      //   current_m05_bucket より前の完成バケットを、g_last_finalized_m05_bucket の
      //   次から順に CopyTicksRange で再構築して g_m05_bars に push する。
      //   - 完成バケットは過去なので全 tick が .tkc に同期済み (A-1 の境界直後
      //     未同期リスクを回避)。
      //   - 学習側 warmup は「最後のバケットは形成中なので捨てる」(SSoT) =
      //     1 バケット遅延確定。live も同じく形成中を捨てることで train-serve 整合。
      //   - OnTimer が呼ばれない隙間 (silent 期間後など) があっても、while で
      //     未確定の完成バケットを全て順次埋める (歯抜け防止)。空バケット
      //     (tick 0) は BuildM05BarFromTicks が false を返し、push せずスキップ
      //     (学習側 filter(tick_count>0) と整合)。ただし時刻基準は前進させる。
      if(g_last_finalized_m05_bucket > 0)
      {
         // 確定対象 = (g_last_finalized_m05_bucket + 30) 〜 (current_m05_bucket - 30) の各完成バケット
         datetime b = g_last_finalized_m05_bucket + 30;
         int guard = 0;  // 無限ループ保護 (最大 2880 本 = 1 日分)
         while(b < current_m05_bucket && guard < 2880)
         {
            M05Bar reb;
            if(BuildM05BarFromTicks(b, reb))
               PushM05Bar(reb);
            // tick 0 の空バケットは push しないが、時刻基準は必ず前進させる
            // (= 学習側で filter(tick_count>0) により削除されるバーと同じ扱い)
            g_last_finalized_m05_bucket = b;
            b += 30;
            guard++;
         }
      }
      else
      {
         // [問題2 修正] 初回の基準点設定。
         // 直前の完成バケット (current_m05_bucket - 30) を「未確定の最新」 として
         // 残すため、基準点は「その 1 つ前 (current_m05_bucket - 60)」 に置く。
         // こうすると次の OnTimer で b = (current-60)+30 = current-30 から while が
         // 回り、直前完成バケットが取りこぼされず確定される。
         // (旧実装は current-30 を基準点にしていたため、直前完成バケット 1 本が
         //  push されないまま「確定済み」 扱いになり 1 本欠落していた。)
         // current_m05_bucket - 60 が負/0 になる極端な起動直後のみガード。
         if(current_m05_bucket >= 60)
            g_last_finalized_m05_bucket = current_m05_bucket - 60;
         else
            g_last_finalized_m05_bucket = current_m05_bucket;  // 退避 (ほぼ起こらない)
      }

      if(current_m3 > g_last_m3_bucket)
      {
         g_last_m3_bucket = current_m3;
         int buf_size = ArraySize(g_m05_bars);
         if(buf_size > 0)
         {
            // ─────────────────────────────────────────────────────────────
            // [Phase 9d 発見 #61] 直近 6 本の M0.5 を一括送信
            // ─────────────────────────────────────────────────────────────
            // 旧: g_m05_bars[buf_size - 1] の 1 本のみ送信
            //   → Python 側 m05_dataframe が steady-state で sparse 化し、
            //     M3 OHLCV が「最後の 30 秒の M0.5 1 本だけから derived」される
            //     構造的バグ。学習側 s1_1_B (full 6 本集約) と乖離していた。
            //
            // 新: 直近 6 本の M0.5 を JSON 配列で一括送信。
            //   → Python 側は順次 process_new_m05_bar に流して m05_dataframe を
            //     正しく充填。M3 集約が学習側と数学的に完全等価になる。
            //
            // 案 X + 方針 2 (Python 側 timestamp ベース close 検知) との併用で
            // 構造的乖離 #61 + 3 分遅延 #62 を同時解消。
            // ─────────────────────────────────────────────────────────────
            int n_bars_to_send = (buf_size < 6) ? buf_size : 6;
            int start_idx = buf_size - n_bars_to_send;

            long spread_points = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
            double notify_spread_pips = (double)spread_points / 10.0;

            // JSON 配列を構築
            string bars_json = "";
            for(int i = 0; i < n_bars_to_send; i++)
            {
               M05Bar bar = g_m05_bars[start_idx + i];
               if(i > 0) bars_json += ",";
               bars_json += StringFormat(
                  "{\"time\":%I64d,\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%I64d}",
                  bar.time,
                  bar.open,
                  bar.high,
                  bar.low,
                  bar.close,
                  bar.volume
               );
            }

            string notify = StringFormat(
               "{\"spread\":%.1f,\"bars\":[%s]}",
               notify_spread_pips,
               bars_json
            );
            ZmqMsg notifyMsg(notify);
            g_m3_notify_socket.send(notifyMsg);

            M05Bar first_bar = g_m05_bars[start_idx];
            M05Bar last_bar = g_m05_bars[buf_size - 1];
            PrintFormat("📡 M3確定通知送信(Timer): M3_bucket=%s, M0.5 range=%s〜%s (%d bars)",
                        TimeToString(current_m3, TIME_DATE|TIME_SECONDS),
                        TimeToString(first_bar.time, TIME_DATE|TIME_SECONDS),
                        TimeToString(last_bar.time, TIME_DATE|TIME_SECONDS),
                        n_bars_to_send);
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

         // [TICK-AGG-FIX A-2 / 二重生成防止] ②g_m05_bars が存在する場合、
         // ①の取得上限 (time_to) を ②の最古バケット時刻にクランプする。
         //
         // A-2 化により ① (CopyTicksRange 過去分) と ② (g_m05_bars = OnTimer が
         // CopyTicksRange で構築した確定バー) が同一の集約ロジックになった結果、
         // クランプしないと ①の [time_from, time_to] と ②のバケット時刻範囲が
         // 重複し、g_history_data に同一時刻 M0.5 が 2 行入る (= §11.34 で潰した
         // 二重 append と同型の汚染が warmup 経路で再発する)。
         // ②の最古バケット未満だけを①が生成し、②がその先を担うことで、
         // 連結後の時刻が重複なく単調増加になる。
         if(m05_realtime_count > 0)
         {
            datetime oldest_rt_bucket = g_m05_bars[0].time;
            if(time_to > oldest_rt_bucket)
               time_to = oldest_rt_bucket;  // ②最古でクランプ (重複帯を①から除外)
         }

         // クランプの結果 time_from >= time_to なら①は不要 (②だけで足りる)
         if(time_from >= time_to)
         {
            PrintFormat("  -> ①過去分スキップ: ②(realtime %d本)が必要範囲を充足", m05_realtime_count);
         }
         else
         {
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
            // [TICK-AGG-FIX A-2] time_to を②最古でクランプ済みのため、ここで最後の
            // バケットを捨てても①の末尾と②の先頭が連続する (重複も歯抜けも出ない)。

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
         }  // end else (time_from < time_to)
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

      // [TICK-AGG-FIX A-2 / 保険] ①②連結後の単調増加 dedup (多層防御)。
      // 上の time_to クランプで重複は構造的に出ないはずだが、①の末尾バケットと
      // ②の先頭が同一時刻になる端数ケースや、②内の歯抜けに備えた最終防御線。
      // rfe 側 m05_dataframe の単調増加ガードと同じ思想。時刻が直前以下のバーを
      // in-place compaction で除去し、g_history_data を厳密に昇順・重複なしにする。
      // (g_history_data は ① resample が昇順 + ② g_m05_bars が昇順蓄積のため、
      //  全体は概ね昇順。ここでは「直前 time 以下」を弾くだけで重複・逆転を排除。)
      {
         int n_before = ArraySize(g_history_data);
         if(n_before > 1)
         {
            int write_idx = 1;  // 先頭[0]は常に保持
            datetime prev_t = g_history_data[0].time;
            for(int r = 1; r < n_before; r++)
            {
               if(g_history_data[r].time > prev_t)
               {
                  if(write_idx != r)
                     g_history_data[write_idx] = g_history_data[r];
                  prev_t = g_history_data[write_idx].time;
                  write_idx++;
               }
               // time <= prev_t のバーは重複/逆転としてスキップ (上書きされる)
            }
            if(write_idx < n_before)
            {
               ArrayResize(g_history_data, write_idx);
               PrintFormat("  -> [dedup] 連結後 %d → %d 本 (重複/逆転 %d 本除去)",
                           n_before, write_idx, n_before - write_idx);
            }
         }
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
