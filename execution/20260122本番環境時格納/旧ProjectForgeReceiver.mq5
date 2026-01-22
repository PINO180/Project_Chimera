//+------------------------------------------------------------------+
//|                                      ProjectForgeReceiver_v2.mq5 |
//|                        Project Forge - 高信頼性取引実行システム |
//|                  レイジー・パイレートと双方向ハートビート対応   |
//+------------------------------------------------------------------+
#property copyright "Project Forge"
#property link      "https://github.com/project-forge"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include <JAson.mqh>  // JSON処理ライブラリ（要インストール）

// ZeroMQ定数（ZeroMQ MQL5バインディング使用）
#include <Zmq/Zmq.mqh>

//--- 入力パラメータ
input string   TradeEndpoint = "tcp://*:5555";      // 取引コマンド受信エンドポイント
input string   HeartbeatEndpoint = "tcp://*:5556";  // ハートビート通信エンドポイント
input double   MaxSlippage = 3.0;                    // 最大スリッページ（pips）
input int      MagicNumber = 20250101;               // マジックナンバー
input bool     EnableLogging = true;                 // 詳細ログ出力
input bool     DryRun = false;                       // ドライラン（実際の発注なし）

//--- グローバル変数
Context context;
Socket tradeSocket;
Socket heartbeatSocket;
CTrade trade;

datetime lastHeartbeatReceived;
int heartbeatTimeout = 30;  // 秒

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("========================================");
   Print("Project Forge Receiver v2.0 初期化開始");
   Print("========================================");
   
   // トレードオブジェクトの設定
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints((int)(MaxSlippage * 10));
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   trade.SetAsyncMode(false);
   
   // ZeroMQコンテキスト作成
   context = new Context("ProjectForge");
   
   // 取引コマンド用REPソケット（リクエスト-レスポンスパターン）
   tradeSocket = context.socket(ZMQ_REP);
   
   if(!tradeSocket.bind(TradeEndpoint))
   {
      Print("✗ 取引ソケットのバインドに失敗: ", TradeEndpoint);
      return INIT_FAILED;
   }
   
   Print("✓ 取引ソケットをバインド: ", TradeEndpoint);
   
   // ハートビート用ROUTERソケット
   heartbeatSocket = context.socket(ZMQ_ROUTER);
   
   if(!heartbeatSocket.bind(HeartbeatEndpoint))
   {
      Print("✗ ハートビートソケットのバインドに失敗: ", HeartbeatEndpoint);
      tradeSocket.unbind(TradeEndpoint);
      return INIT_FAILED;
   }
   
   Print("✓ ハートビートソケットをバインド: ", HeartbeatEndpoint);
   
   lastHeartbeatReceived = TimeCurrent();
   
   if(DryRun)
   {
      Print("⚠ ドライランモード: 実際の発注は行いません");
   }
   
   Print("========================================");
   Print("✓ Project Forge Receiver v2.0 起動完了");
   Print("========================================");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("========================================");
   Print("Project Forge Receiver v2.0 終了処理");
   Print("========================================");
   
   // ソケットのクローズ
   if(tradeSocket != NULL)
   {
      tradeSocket.unbind(TradeEndpoint);
      tradeSocket.disconnect(TradeEndpoint);
      delete tradeSocket;
   }
   
   if(heartbeatSocket != NULL)
   {
      heartbeatSocket.unbind(HeartbeatEndpoint);
      heartbeatSocket.disconnect(HeartbeatEndpoint);
      delete heartbeatSocket;
   }
   
   // コンテキストのクリーンアップ
   if(context != NULL)
   {
      delete context;
   }
   
   Print("✓ クリーンアップ完了");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // 取引コマンドの受信チェック（ノンブロッキング）
   CheckTradeCommands();
   
   // ハートビートの受信チェック
   CheckHeartbeat();
   
   // ハートビートタイムアウトチェック
   if(TimeCurrent() - lastHeartbeatReceived > heartbeatTimeout)
   {
      Print("⚠ 警告: Pythonコアからのハートビートが途絶えています（",
            (TimeCurrent() - lastHeartbeatReceived), "秒）");
   }
}

//+------------------------------------------------------------------+
//| 取引コマンドの受信と処理                                        |
//+------------------------------------------------------------------+
void CheckTradeCommands()
{
   ZmqMsg request;
   
   // ノンブロッキング受信
   if(tradeSocket.recv(request, true))
   {
      string jsonMessage = request.getData();
      
      if(EnableLogging)
      {
         Print("受信メッセージ: ", jsonMessage);
      }
      
      // JSON解析
      CJAVal json;
      if(!json.Deserialize(jsonMessage))
      {
         Print("✗ JSON解析失敗");
         SendResponse("NACK", "JSON解析エラー");
         return;
      }
      
      // メッセージタイプの判定
      string messageType = json["message_type"].ToStr();
      
      if(messageType == "TRADE_COMMAND")
      {
         ProcessTradeCommand(json);
      }
      else if(messageType == "REQUEST_BROKER_STATE")
      {
         SendBrokerState();
      }
      else
      {
         Print("✗ 未知のメッセージタイプ: ", messageType);
         SendResponse("NACK", "未知のメッセージタイプ");
      }
   }
}

//+------------------------------------------------------------------+
//| 取引コマンドの処理                                              |
//+------------------------------------------------------------------+
void ProcessTradeCommand(CJAVal &json)
{
   string action = json["action"].ToStr();
   double lots = json["lots"].ToDbl();
   double entryPrice = json["entry_price"].ToDbl();
   double stopLoss = json["stop_loss"].ToDbl();
   double takeProfit = json["take_profit"].ToDbl();
   double confidence = json["confidence_m2"].ToDbl();
   int messageId = (int)json["message_id"].ToInt();
   
   Print("========================================");
   Print("取引コマンド受信 [#", messageId, "]");
   Print("========================================");
   Print("アクション: ", action);
   Print("ロット: ", lots);
   Print("エントリー: ", entryPrice);
   Print("損切り: ", stopLoss);
   Print("利食い: ", takeProfit);
   Print("確信度: ", confidence * 100, "%");
   
   // HOLDコマンドの場合
   if(action == "HOLD")
   {
      Print("→ アクション不要（HOLD）");
      SendResponse("ACK", "HOLD確認");
      return;
   }
   
   // ドライランモード
   if(DryRun)
   {
      Print("⚠ ドライランモード: 発注をシミュレート");
      SendResponse("ACK", "ドライラン確認");
      return;
   }
   
   // 実際の発注処理
   bool success = false;
   
   if(action == "BUY")
   {
      success = trade.Buy(lots, Symbol(), 0, stopLoss, takeProfit, 
                         "Forge:" + IntegerToString(messageId));
   }
   else if(action == "SELL")
   {
      success = trade.Sell(lots, Symbol(), 0, stopLoss, takeProfit,
                          "Forge:" + IntegerToString(messageId));
   }
   else
   {
      Print("✗ 無効なアクション: ", action);
      SendResponse("NACK", "無効なアクション");
      return;
   }
   
   // 結果の確認
   if(success)
   {
      Print("✓ 発注成功");
      Print("  チケット番号: ", trade.ResultOrder());
      Print("  約定価格: ", trade.ResultPrice());
      
      SendResponse("ACK", "発注成功");
   }
   else
   {
      Print("✗ 発注失敗");
      Print("  エラーコード: ", trade.ResultRetcode());
      Print("  エラー説明: ", trade.ResultRetcodeDescription());
      
      SendResponse("NACK", "発注失敗: " + trade.ResultRetcodeDescription());
   }
   
   Print("========================================");
}

//+------------------------------------------------------------------+
//| ブローカー状態の送信                                            |
//+------------------------------------------------------------------+
void SendBrokerState()
{
   Print("ブローカー状態リクエストを受信");
   
   CJAVal response;
   response["message_type"] = "BROKER_STATE_RESPONSE";
   response["timestamp"] = TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);
   
   // 口座情報
   CJAVal data;
   data["equity"] = AccountInfoDouble(ACCOUNT_EQUITY);
   data["balance"] = AccountInfoDouble(ACCOUNT_BALANCE);
   data["margin"] = AccountInfoDouble(ACCOUNT_MARGIN);
   data["free_margin"] = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   
   // ポジション情報
   CJAVal positions;
   int totalPositions = PositionsTotal();
   
   for(int i = 0; i < totalPositions; i++)
   {
      ulong ticket = PositionGetTicket(i);
      
      if(PositionSelectByTicket(ticket))
      {
         // このEAのポジションのみ
         if(PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            CJAVal pos;
            pos["ticket"] = (long)ticket;
            pos["symbol"] = PositionGetString(POSITION_SYMBOL);
            pos["direction"] = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? "BUY" : "SELL";
            pos["lots"] = PositionGetDouble(POSITION_VOLUME);
            pos["entry_price"] = PositionGetDouble(POSITION_PRICE_OPEN);
            pos["stop_loss"] = PositionGetDouble(POSITION_SL);
            pos["take_profit"] = PositionGetDouble(POSITION_TP);
            pos["entry_time"] = TimeToString((datetime)PositionGetInteger(POSITION_TIME), TIME_DATE|TIME_SECONDS);
            pos["unrealized_pnl"] = PositionGetDouble(POSITION_PROFIT);
            
            positions.Add(pos);
         }
      }
   }
   
   data["positions"] = positions;
   response["data"] = data;
   
   // JSONシリアライズと送信
   string jsonResponse = response.Serialize();
   
   ZmqMsg responseMsg(jsonResponse);
   tradeSocket.send(responseMsg);
   
   Print("✓ ブローカー状態を送信しました");
}

//+------------------------------------------------------------------+
//| レスポンス送信（ACK/NACK）                                      |
//+------------------------------------------------------------------+
void SendResponse(string messageType, string message)
{
   CJAVal response;
   response["message_type"] = messageType;
   response["message"] = message;
   response["timestamp"] = TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);
   
   string jsonResponse = response.Serialize();
   
   ZmqMsg responseMsg(jsonResponse);
   tradeSocket.send(responseMsg);
   
   if(EnableLogging)
   {
      Print("レスポンス送信: ", messageType, " - ", message);
   }
}

//+------------------------------------------------------------------+
//| ハートビートの受信と処理                                        |
//+------------------------------------------------------------------+
void CheckHeartbeat()
{
   ZmqMsg frames[];
   
   // マルチパートメッセージ受信（ノンブロッキング）
   int numFrames = heartbeatSocket.recvMulti(frames, true);
   
   if(numFrames > 0)
   {
      // ROUTERソケット: [identity, delimiter, payload]の形式
      if(numFrames >= 3)
      {
         string identity = frames[0].getData();
         string payload = frames[2].getData();
         
         // JSON解析
         CJAVal json;
         if(json.Deserialize(payload))
         {
            string messageType = json["message_type"].ToStr();
            
            if(messageType == "PING")
            {
               lastHeartbeatReceived = TimeCurrent();
               
               if(EnableLogging)
               {
                  Print("PING受信 from ", identity);
               }
               
               // PONGを返信
               SendPong(identity);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| PONGメッセージの送信                                            |
//+------------------------------------------------------------------+
void SendPong(string identity)
{
   CJAVal pong;
   pong["message_type"] = "PONG";
   pong["timestamp"] = TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);
   
   string jsonPong = pong.Serialize();
   
   // マルチパートメッセージ送信: [identity, delimiter, payload]
   ZmqMsg frames[3];
   frames[0].setData(identity);
   frames[1].setData("");  // デリミタ（空フレーム）
   frames[2].setData(jsonPong);
   
   heartbeatSocket.sendMulti(frames);
   
   if(EnableLogging)
   {
      Print("PONG送信 to ", identity);
   }
}

//+------------------------------------------------------------------+
//| チャート上の情報表示                                            |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
   // チャートイベント処理（必要に応じて実装）
}

//+------------------------------------------------------------------+