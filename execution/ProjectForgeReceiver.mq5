//+------------------------------------------------------------------+
//|                                          ProjectForgeReceiver.mq5|
//|                                  Project Forge - AI Trading Core |
//|                                                                    |
//| ZeroMQを介してPythonから送信される取引シグナルを受信し、           |
//| MetaTrader 5で実際の取引注文を実行するExpert Advisor             |
//+------------------------------------------------------------------+
#property copyright "Project Forge"
#property link      ""
#property version   "2.00"
#property strict

// ZeroMQライブラリのインクルード
#include <Zmq/Zmq.mqh>

// 【重要】信頼性の高いJSONパーサーライブラリをインクルード
// MQL5 CodeBaseから取得: https://www.mql5.com/en/code/13663
#include <JAson.mqh>

// 入力パラメータ
input string   ZMQ_ENDPOINT = "tcp://127.0.0.1:5555";  // ZeroMQエンドポイント
input int      MAGIC_NUMBER = 20250102;                // マジックナンバー
input double   MAX_SLIPPAGE = 3.0;                     // 最大スリッページ（ポイント）
input bool     ENABLE_LOGGING = true;                  // ログ出力の有効化
input string   LOG_FILE_PREFIX = "ProjectForge_";      // ログファイル接頭辞

// グローバル変数
Context context;                  // ZeroMQコンテキスト
Socket  subscriber;               // ZeroMQサブスクライバーソケット
bool    zmq_initialized = false;  // ZeroMQ初期化フラグ
int     messages_received = 0;    // 受信メッセージカウンタ
int     orders_executed = 0;      // 執行注文カウンタ
int     orders_failed = 0;        // 失敗注文カウンタ
datetime last_message_time = 0;   // 最終メッセージ受信時刻

//+------------------------------------------------------------------+
//| Expert Advisor 初期化関数                                         |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("Project Forge Receiver EA v2.0 を起動中...");
    Print("【重要】JAson.mqhライブラリを使用した安全なJSON解析");
    Print("========================================");
    
    // ZeroMQの初期化
    if(!InitializeZMQ())
    {
        Print("エラー: ZeroMQの初期化に失敗しました。");
        return(INIT_FAILED);
    }
    
    Print("✓ ZeroMQ初期化成功");
    Print("エンドポイント: ", ZMQ_ENDPOINT);
    Print("マジックナンバー: ", MAGIC_NUMBER);
    Print("========================================");
    Print("AIシグナル待機中...");
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert Advisor 終了関数                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("========================================");
    Print("Project Forge Receiver EA を終了中...");
    Print("統計情報:");
    Print("  - 受信メッセージ数: ", messages_received);
    Print("  - 執行成功: ", orders_executed);
    Print("  - 執行失敗: ", orders_failed);
    if(orders_executed + orders_failed > 0)
    {
        double success_rate = (double)orders_executed / (orders_executed + orders_failed) * 100.0;
        Print("  - 成功率: ", DoubleToString(success_rate, 2), "%");
    }
    Print("========================================");
    
    // ZeroMQソケットとコンテキストのクリーンアップ
    CleanupZMQ();
}

//+------------------------------------------------------------------+
//| Tick イベント処理関数                                            |
//+------------------------------------------------------------------+
void OnTick()
{
    // ZeroMQが初期化されていない場合は何もしない
    if(!zmq_initialized)
        return;
    
    // メッセージを非ブロッキングで受信
    string message = "";
    if(ReceiveMessage(message))
    {
        messages_received++;
        last_message_time = TimeCurrent();
        
        // メッセージを解析
        if(ENABLE_LOGGING)
            LogMessage("受信", message);
        
        // JSONを解析して取引コマンドを実行
        ProcessTradeCommand(message);
    }
}

//+------------------------------------------------------------------+
//| ZeroMQ初期化                                                      |
//+------------------------------------------------------------------+
bool InitializeZMQ()
{
    // コンテキスト作成
    if(!context.initialize())
    {
        Print("ZeroMQコンテキストの初期化に失敗しました。");
        return false;
    }
    
    // SUBソケット作成
    if(!subscriber.initialize(context, ZMQ_SUB))
    {
        Print("ZeroMQソケットの初期化に失敗しました。");
        context.destroy();
        return false;
    }
    
    // すべてのメッセージを購読（フィルタなし）
    if(!subscriber.subscribe(""))
    {
        Print("ZeroMQ購読の設定に失敗しました。");
        subscriber.destroy();
        context.destroy();
        return false;
    }
    
    // エンドポイントに接続
    if(!subscriber.connect(ZMQ_ENDPOINT))
    {
        Print("ZeroMQエンドポイントへの接続に失敗しました: ", ZMQ_ENDPOINT);
        subscriber.destroy();
        context.destroy();
        return false;
    }
    
    // 接続安定化のための待機
    Sleep(500);
    
    zmq_initialized = true;
    return true;
}

//+------------------------------------------------------------------+
//| ZeroMQクリーンアップ                                              |
//+------------------------------------------------------------------+
void CleanupZMQ()
{
    if(zmq_initialized)
    {
        subscriber.disconnect(ZMQ_ENDPOINT);
        subscriber.destroy();
        context.destroy();
        zmq_initialized = false;
        Print("✓ ZeroMQクリーンアップ完了");
    }
}

//+------------------------------------------------------------------+
//| メッセージ受信（非ブロッキング）                                   |
//+------------------------------------------------------------------+
bool ReceiveMessage(string &message)
{
    ZmqMsg msg;
    
    // 非ブロッキングで受信を試みる
    int result = subscriber.recv(msg, ZMQ_DONTWAIT);
    
    if(result > 0)
    {
        // メッセージを取得
        message = msg.getData();
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| 取引コマンド処理（JAson.mqhライブラリを使用）                      |
//+------------------------------------------------------------------+
void ProcessTradeCommand(string json_message)
{
    // 【改善】JAson.mqhライブラリでJSON解析
    CJAVal json_parser;
    
    if(!json_parser.Deserialize(json_message))
    {
        Print("エラー: JSON解析に失敗しました。");
        Print("受信メッセージ: ", json_message);
        orders_failed++;
        if(ENABLE_LOGGING)
            LogMessage("ERROR", "JSON解析失敗: " + json_message);
        return;
    }
    
    // 必須フィールドの存在確認
    if(!json_parser.FindKey("action"))
    {
        Print("エラー: 'action'フィールドが見つかりません。");
        orders_failed++;
        return;
    }
    
    // 値を安全に取得
    string action = json_parser["action"].ToStr();
    double lots = json_parser.FindKey("lots") ? json_parser["lots"].ToDbl() : 0.0;
    double entry_price = json_parser.FindKey("entry_price") ? json_parser["entry_price"].ToDbl() : 0.0;
    double stop_loss = json_parser.FindKey("stop_loss") ? json_parser["stop_loss"].ToDbl() : 0.0;
    double take_profit = json_parser.FindKey("take_profit") ? json_parser["take_profit"].ToDbl() : 0.0;
    double confidence = json_parser.FindKey("confidence") ? json_parser["confidence"].ToDbl() : 0.0;
    
    // アクション判定
    if(action == "HOLD")
    {
        if(ENABLE_LOGGING)
            Print("シグナル: HOLD - エントリー見送り");
        return;
    }
    
    // ロット数の検証
    if(lots <= 0)
    {
        Print("エラー: 無効なロット数 (", lots, ")");
        orders_failed++;
        return;
    }
    
    // 取引実行
    bool success = false;
    if(action == "BUY")
    {
        success = ExecuteBuyOrder(lots, stop_loss, take_profit, confidence);
    }
    else if(action == "SELL")
    {
        success = ExecuteSellOrder(lots, stop_loss, take_profit, confidence);
    }
    else
    {
        Print("エラー: 不明なアクション - ", action);
        orders_failed++;
        return;
    }
    
    // 統計更新
    if(success)
        orders_executed++;
    else
        orders_failed++;
}

//+------------------------------------------------------------------+
//| 買い注文実行                                                      |
//+------------------------------------------------------------------+
bool ExecuteBuyOrder(double lots, double sl, double tp, double confidence)
{
    MqlTradeRequest request;
    MqlTradeResult  result;
    
    // 現在のシンボル情報を取得
    string symbol = _Symbol;
    double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    
    // リクエスト構造体の初期化
    ZeroMemory(request);
    ZeroMemory(result);
    
    // 注文パラメータの設定
    request.action    = TRADE_ACTION_DEAL;              // 即座に取引
    request.symbol    = symbol;                         // シンボル
    request.volume    = lots;                           // ロット数
    request.type      = ORDER_TYPE_BUY;                 // 買い注文
    request.price     = ask;                            // エントリー価格
    request.sl        = NormalizeDouble(sl, _Digits);   // 損切り
    request.tp        = NormalizeDouble(tp, _Digits);   // 利食い
    request.deviation = (ulong)MAX_SLIPPAGE;            // スリッページ
    request.magic     = MAGIC_NUMBER;                   // マジックナンバー
    request.comment   = StringFormat("AI_BUY_%.2f%%", confidence * 100);
    request.type_filling = ORDER_FILLING_FOK;           // Fill or Kill
    
    // 注文送信
    bool sent = OrderSend(request, result);
    
    if(sent && result.retcode == TRADE_RETCODE_DONE)
    {
        Print("✓ 買い注文成功: ", lots, "ロット @ ", ask, " (SL:", sl, " TP:", tp, ")");
        Print("  チケット番号: ", result.order);
        Print("  確信度: ", DoubleToString(confidence * 100, 2), "%");
        
        if(ENABLE_LOGGING)
            LogOrderSuccess("BUY", lots, ask, sl, tp, result.order);
        
        return true;
    }
    else
    {
        Print("✗ 買い注文失敗: ", GetRetcodeDescription(result.retcode));
        Print("  エラーコード: ", result.retcode);
        
        if(ENABLE_LOGGING)
            LogOrderFailure("BUY", lots, ask, result.retcode);
        
        return false;
    }
}

//+------------------------------------------------------------------+
//| 売り注文実行                                                      |
//+------------------------------------------------------------------+
bool ExecuteSellOrder(double lots, double sl, double tp, double confidence)
{
    MqlTradeRequest request;
    MqlTradeResult  result;
    
    // 現在のシンボル情報を取得
    string symbol = _Symbol;
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    
    // リクエスト構造体の初期化
    ZeroMemory(request);
    ZeroMemory(result);
    
    // 注文パラメータの設定
    request.action    = TRADE_ACTION_DEAL;              // 即座に取引
    request.symbol    = symbol;                         // シンボル
    request.volume    = lots;                           // ロット数
    request.type      = ORDER_TYPE_SELL;                // 売り注文
    request.price     = bid;                            // エントリー価格
    request.sl        = NormalizeDouble(sl, _Digits);   // 損切り
    request.tp        = NormalizeDouble(tp, _Digits);   // 利食い
    request.deviation = (ulong)MAX_SLIPPAGE;            // スリッページ
    request.magic     = MAGIC_NUMBER;                   // マジックナンバー
    request.comment   = StringFormat("AI_SELL_%.2f%%", confidence * 100);
    request.type_filling = ORDER_FILLING_FOK;           // Fill or Kill
    
    // 注文送信
    bool sent = OrderSend(request, result);
    
    if(sent && result.retcode == TRADE_RETCODE_DONE)
    {
        Print("✓ 売り注文成功: ", lots, "ロット @ ", bid, " (SL:", sl, " TP:", tp, ")");
        Print("  チケット番号: ", result.order);
        Print("  確信度: ", DoubleToString(confidence * 100, 2), "%");
        
        if(ENABLE_LOGGING)
            LogOrderSuccess("SELL", lots, bid, sl, tp, result.order);
        
        return true;
    }
    else
    {
        Print("✗ 売り注文失敗: ", GetRetcodeDescription(result.retcode));
        Print("  エラーコード: ", result.retcode);
        
        if(ENABLE_LOGGING)
            LogOrderFailure("SELL", lots, bid, result.retcode);
        
        return false;
    }
}

//+------------------------------------------------------------------+
//| リターンコード説明                                                |
//+------------------------------------------------------------------+
string GetRetcodeDescription(uint retcode)
{
    switch(retcode)
    {
        case TRADE_RETCODE_DONE:           return "注文完了";
        case TRADE_RETCODE_PLACED:         return "注文受付";
        case TRADE_RETCODE_DONE_PARTIAL:   return "部分約定";
        case TRADE_RETCODE_ERROR:          return "一般エラー";
        case TRADE_RETCODE_TIMEOUT:        return "タイムアウト";
        case TRADE_RETCODE_INVALID:        return "無効なリクエスト";
        case TRADE_RETCODE_INVALID_VOLUME: return "無効なボリューム";
        case TRADE_RETCODE_INVALID_PRICE:  return "無効な価格";
        case TRADE_RETCODE_INVALID_STOPS:  return "無効なストップ";
        case TRADE_RETCODE_TRADE_DISABLED: return "取引無効";
        case TRADE_RETCODE_MARKET_CLOSED:  return "市場クローズ";
        case TRADE_RETCODE_NO_MONEY:       return "証拠金不足";
        case TRADE_RETCODE_PRICE_OFF:      return "価格変更";
        case TRADE_RETCODE_REJECT:         return "リクエスト拒否";
        default:                           return "不明なエラー";
    }
}

//+------------------------------------------------------------------+
//| ログ出力: メッセージ受信                                          |
//+------------------------------------------------------------------+
void LogMessage(string status, string message)
{
    string filename = LOG_FILE_PREFIX + TimeToString(TimeCurrent(), TIME_DATE) + ".log";
    int handle = FileOpen(filename, FILE_READ|FILE_WRITE|FILE_TXT|FILE_ANSI, '\t');
    
    if(handle != INVALID_HANDLE)
    {
        FileSeek(handle, 0, SEEK_END);
        string log_entry = StringFormat("[%s] %s: %s\n", 
                                       TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
                                       status,
                                       message);
        FileWriteString(handle, log_entry);
        FileClose(handle);
    }
}

//+------------------------------------------------------------------+
//| ログ出力: 注文成功                                                |
//+------------------------------------------------------------------+
void LogOrderSuccess(string action, double lots, double price, double sl, double tp, ulong ticket)
{
    string log_line = StringFormat("ORDER_SUCCESS | %s | Lots:%.2f | Price:%.5f | SL:%.5f | TP:%.5f | Ticket:%I64u",
                                   action, lots, price, sl, tp, ticket);
    LogMessage("TRADE", log_line);
}

//+------------------------------------------------------------------+
//| ログ出力: 注文失敗                                                |
//+------------------------------------------------------------------+
void LogOrderFailure(string action, double lots, double price, uint retcode)
{
    string log_line = StringFormat("ORDER_FAILED | %s | Lots:%.2f | Price:%.5f | Error:%u (%s)",
                                   action, lots, price, retcode, GetRetcodeDescription(retcode));
    LogMessage("ERROR", log_line);
}
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Expert Advisor 終了関数                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("========================================");
    Print("Project Forge Receiver EA を終了中...");
    Print("統計情報:");
    Print("  - 受信メッセージ数: ", messages_received);
    Print("  - 執行成功: ", orders_executed);
    Print("  - 執行失敗: ", orders_failed);
    Print("========================================");
    
    // ZeroMQソケットとコンテキストのクリーンアップ
    CleanupZMQ();
}

//+------------------------------------------------------------------+
//| Tick イベント処理関数                                            |
//+------------------------------------------------------------------+
void OnTick()
{
    // ZeroMQが初期化されていない場合は何もしない
    if(!zmq_initialized)
        return;
    
    // メッセージを非ブロッキングで受信
    string message = "";
    if(ReceiveMessage(message))
    {
        messages_received++;
        last_message_time = TimeCurrent();
        
        // メッセージを解析
        if(ENABLE_LOGGING)
            LogMessage("受信", message);
        
        // JSONを解析して取引コマンドを実行
        ProcessTradeCommand(message);
    }
}

//+------------------------------------------------------------------+
//| ZeroMQ初期化                                                      |
//+------------------------------------------------------------------+
bool InitializeZMQ()
{
    // コンテキスト作成
    if(!context.initialize())
    {
        Print("ZeroMQコンテキストの初期化に失敗しました。");
        return false;
    }
    
    // SUBソケット作成
    if(!subscriber.initialize(context, ZMQ_SUB))
    {
        Print("ZeroMQソケットの初期化に失敗しました。");
        context.destroy();
        return false;
    }
    
    // すべてのメッセージを購読（フィルタなし）
    if(!subscriber.subscribe(""))
    {
        Print("ZeroMQ購読の設定に失敗しました。");
        subscriber.destroy();
        context.destroy();
        return false;
    }
    
    // エンドポイントに接続
    if(!subscriber.connect(ZMQ_ENDPOINT))
    {
        Print("ZeroMQエンドポイントへの接続に失敗しました: ", ZMQ_ENDPOINT);
        subscriber.destroy();
        context.destroy();
        return false;
    }
    
    // 接続安定化のための待機
    Sleep(500);
    
    zmq_initialized = true;
    return true;
}

//+------------------------------------------------------------------+
//| ZeroMQクリーンアップ                                              |
//+------------------------------------------------------------------+
void CleanupZMQ()
{
    if(zmq_initialized)
    {
        subscriber.disconnect(ZMQ_ENDPOINT);
        subscriber.destroy();
        context.destroy();
        zmq_initialized = false;
        Print("✓ ZeroMQクリーンアップ完了");
    }
}

//+------------------------------------------------------------------+
//| メッセージ受信（非ブロッキング）                                   |
//+------------------------------------------------------------------+
bool ReceiveMessage(string &message)
{
    ZmqMsg msg;
    
    // 非ブロッキングで受信を試みる
    int result = subscriber.recv(msg, ZMQ_DONTWAIT);
    
    if(result > 0)
    {
        // メッセージを取得
        message = msg.getData();
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| 取引コマンド処理                                                  |
//+------------------------------------------------------------------+
void ProcessTradeCommand(string json_message)
{
    // JSON解析（簡易版 - 実際にはJSON解析ライブラリを使用すべき）
    string action = ExtractJsonString(json_message, "action");
    double lots = ExtractJsonDouble(json_message, "lots");
    double entry_price = ExtractJsonDouble(json_message, "entry_price");
    double stop_loss = ExtractJsonDouble(json_message, "stop_loss");
    double take_profit = ExtractJsonDouble(json_message, "take_profit");
    double confidence = ExtractJsonDouble(json_message, "confidence");
    
    // アクション判定
    if(action == "HOLD")
    {
        if(ENABLE_LOGGING)
            Print("シグナル: HOLD - エントリー見送り");
        return;
    }
    
    // ロット数の検証
    if(lots <= 0)
    {
        Print("エラー: 無効なロット数 (", lots, ")");
        orders_failed++;
        return;
    }
    
    // 取引実行
    bool success = false;
    if(action == "BUY")
    {
        success = ExecuteBuyOrder(lots, stop_loss, take_profit, confidence);
    }
    else if(action == "SELL")
    {
        success = ExecuteSellOrder(lots, stop_loss, take_profit, confidence);
    }
    else
    {
        Print("エラー: 不明なアクション - ", action);
        orders_failed++;
        return;
    }
    
    // 統計更新
    if(success)
        orders_executed++;
    else
        orders_failed++;
}

//+------------------------------------------------------------------+
//| 買い注文実行                                                      |
//+------------------------------------------------------------------+
bool ExecuteBuyOrder(double lots, double sl, double tp, double confidence)
{
    MqlTradeRequest request;
    MqlTradeResult  result;
    
    // 現在のシンボル情報を取得
    string symbol = _Symbol;
    double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    
    // リクエスト構造体の初期化
    ZeroMemory(request);
    ZeroMemory(result);
    
    // 注文パラメータの設定
    request.action    = TRADE_ACTION_DEAL;              // 即座に取引
    request.symbol    = symbol;                         // シンボル
    request.volume    = lots;                           // ロット数
    request.type      = ORDER_TYPE_BUY;                 // 買い注文
    request.price     = ask;                            // エントリー価格
    request.sl        = NormalizeDouble(sl, _Digits);   // 損切り
    request.tp        = NormalizeDouble(tp, _Digits);   // 利食い
    request.deviation = (ulong)MAX_SLIPPAGE;            // スリッページ
    request.magic     = MAGIC_NUMBER;                   // マジックナンバー
    request.comment   = StringFormat("AI_BUY_%.2f%%", confidence * 100);
    request.type_filling = ORDER_FILLING_FOK;           // Fill or Kill
    
    // 注文送信
    bool sent = OrderSend(request, result);
    
    if(sent && result.retcode == TRADE_RETCODE_DONE)
    {
        Print("✓ 買い注文成功: ", lots, "ロット @ ", ask, " (SL:", sl, " TP:", tp, ")");
        Print("  チケット番号: ", result.order);
        Print("  確信度: ", DoubleToString(confidence * 100, 2), "%");
        
        if(ENABLE_LOGGING)
            LogOrderSuccess("BUY", lots, ask, sl, tp, result.order);
        
        return true;
    }
    else
    {
        Print("✗ 買い注文失敗: ", GetRetcodeDescription(result.retcode));
        Print("  エラーコード: ", result.retcode);
        
        if(ENABLE_LOGGING)
            LogOrderFailure("BUY", lots, ask, result.retcode);
        
        return false;
    }
}

//+------------------------------------------------------------------+
//| 売り注文実行                                                      |
//+------------------------------------------------------------------+
bool ExecuteSellOrder(double lots, double sl, double tp, double confidence)
{
    MqlTradeRequest request;
    MqlTradeResult  result;
    
    // 現在のシンボル情報を取得
    string symbol = _Symbol;
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    
    // リクエスト構造体の初期化
    ZeroMemory(request);
    ZeroMemory(result);
    
    // 注文パラメータの設定
    request.action    = TRADE_ACTION_DEAL;              // 即座に取引
    request.symbol    = symbol;                         // シンボル
    request.volume    = lots;                           // ロット数
    request.type      = ORDER_TYPE_SELL;                // 売り注文
    request.price     = bid;                            // エントリー価格
    request.sl        = NormalizeDouble(sl, _Digits);   // 損切り
    request.tp        = NormalizeDouble(tp, _Digits);   // 利食い
    request.deviation = (ulong)MAX_SLIPPAGE;            // スリッページ
    request.magic     = MAGIC_NUMBER;                   // マジックナンバー
    request.comment   = StringFormat("AI_SELL_%.2f%%", confidence * 100);
    request.type_filling = ORDER_FILLING_FOK;           // Fill or Kill
    
    // 注文送信
    bool sent = OrderSend(request, result);
    
    if(sent && result.retcode == TRADE_RETCODE_DONE)
    {
        Print("✓ 売り注文成功: ", lots, "ロット @ ", bid, " (SL:", sl, " TP:", tp, ")");
        Print("  チケット番号: ", result.order);
        Print("  確信度: ", DoubleToString(confidence * 100, 2), "%");
        
        if(ENABLE_LOGGING)
            LogOrderSuccess("SELL", lots, bid, sl, tp, result.order);
        
        return true;
    }
    else
    {
        Print("✗ 売り注文失敗: ", GetRetcodeDescription(result.retcode));
        Print("  エラーコード: ", result.retcode);
        
        if(ENABLE_LOGGING)
            LogOrderFailure("SELL", lots, bid, result.retcode);
        
        return false;
    }
}

//+------------------------------------------------------------------+
//| JSON文字列値抽出（簡易版）                                         |
//+------------------------------------------------------------------+
string ExtractJsonString(string json, string key)
{
    string search_pattern = "\"" + key + "\"";
    int start = StringFind(json, search_pattern);
    
    if(start == -1)
        return "";
    
    // キーの後のコロンを探す
    start = StringFind(json, ":", start);
    if(start == -1)
        return "";
    
    // 値の開始位置を探す（空白と引用符をスキップ）
    start++;
    while(start < StringLen(json) && (StringGetCharacter(json, start) == ' ' || 
                                      StringGetCharacter(json, start) == '\t' ||
                                      StringGetCharacter(json, start) == '\n'))
        start++;
    
    if(StringGetCharacter(json, start) == '\"')
    {
        start++; // 開始引用符をスキップ
        int end = StringFind(json, "\"", start);
        if(end == -1)
            return "";
        return StringSubstr(json, start, end - start);
    }
    
    return "";
}

//+------------------------------------------------------------------+
//| JSON数値抽出（簡易版）                                             |
//+------------------------------------------------------------------+
double ExtractJsonDouble(string json, string key)
{
    string search_pattern = "\"" + key + "\"";
    int start = StringFind(json, search_pattern);
    
    if(start == -1)
        return 0.0;
    
    // キーの後のコロンを探す
    start = StringFind(json, ":", start);
    if(start == -1)
        return 0.0;
    
    // 値の開始位置を探す
    start++;
    while(start < StringLen(json) && (StringGetCharacter(json, start) == ' ' || 
                                      StringGetCharacter(json, start) == '\t'))
        start++;
    
    // 数値文字列を抽出
    string number_str = "";
    while(start < StringLen(json))
    {
        ushort ch = StringGetCharacter(json, start);
        if((ch >= '0' && ch <= '9') || ch == '.' || ch == '-' || ch == '+' || ch == 'e' || ch == 'E')
        {
            number_str += CharToString((uchar)ch);
            start++;
        }
        else
            break;
    }
    
    return StringToDouble(number_str);
}

//+------------------------------------------------------------------+
//| リターンコード説明                                                |
//+------------------------------------------------------------------+
string GetRetcodeDescription(uint retcode)
{
    switch(retcode)
    {
        case TRADE_RETCODE_DONE:           return "注文完了";
        case TRADE_RETCODE_PLACED:         return "注文受付";
        case TRADE_RETCODE_DONE_PARTIAL:   return "部分約定";
        case TRADE_RETCODE_ERROR:          return "一般エラー";
        case TRADE_RETCODE_TIMEOUT:        return "タイムアウト";
        case TRADE_RETCODE_INVALID:        return "無効なリクエスト";
        case TRADE_RETCODE_INVALID_VOLUME: return "無効なボリューム";
        case TRADE_RETCODE_INVALID_PRICE:  return "無効な価格";
        case TRADE_RETCODE_INVALID_STOPS:  return "無効なストップ";
        case TRADE_RETCODE_TRADE_DISABLED: return "取引無効";
        case TRADE_RETCODE_MARKET_CLOSED:  return "市場クローズ";
        case TRADE_RETCODE_NO_MONEY:       return "証拠金不足";
        case TRADE_RETCODE_PRICE_OFF:      return "価格変更";
        case TRADE_RETCODE_REJECT:         return "リクエスト拒否";
        default:                           return "不明なエラー";
    }
}

//+------------------------------------------------------------------+
//| ログ出力: メッセージ受信                                          |
//+------------------------------------------------------------------+
void LogMessage(string status, string message)
{
    string filename = LOG_FILE_PREFIX + TimeToString(TimeCurrent(), TIME_DATE) + ".log";
    int handle = FileOpen(filename, FILE_READ|FILE_WRITE|FILE_TXT|FILE_ANSI, '\t');
    
    if(handle != INVALID_HANDLE)
    {
        FileSeek(handle, 0, SEEK_END);
        string log_entry = StringFormat("[%s] %s: %s\n", 
                                       TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
                                       status,
                                       message);
        FileWriteString(handle, log_entry);
        FileClose(handle);
    }
}

//+------------------------------------------------------------------+
//| ログ出力: 注文成功                                                |
//+------------------------------------------------------------------+
void LogOrderSuccess(string action, double lots, double price, double sl, double tp, ulong ticket)
{
    string log_line = StringFormat("ORDER_SUCCESS | %s | Lots:%.2f | Price:%.5f | SL:%.5f | TP:%.5f | Ticket:%I64u",
                                   action, lots, price, sl, tp, ticket);
    LogMessage("TRADE", log_line);
}

//+------------------------------------------------------------------+
//| ログ出力: 注文失敗                                                |
//+------------------------------------------------------------------+
void LogOrderFailure(string action, double lots, double price, uint retcode)
{
    string log_line = StringFormat("ORDER_FAILED | %s | Lots:%.2f | Price:%.5f | Error:%u (%s)",
                                   action, lots, price, retcode, GetRetcodeDescription(retcode));
    LogMessage("ERROR", log_line);
}
//+------------------------------------------------------------------+