python3 -c "
import csv
bins = [('<=3',0,3),('3-4',3,4),('4-5',4,5),('5-10',5,10),('10-15',10,15),('15-20',15,20),('>20',20,1e9)]
paths = {
 'R2_2.0と0.3 ':'/workspace/data/XAUUSD/stratum_7_models/backtest_simulator_results/M2_20260701_231941_Th0.7_D0.3_R2 2.0と0.3/detailed_trade_log_v5_M2.csv',
 'R2_1.5と0.3 ':'/workspace/data/XAUUSD/stratum_7_models/backtest_simulator_results/M2_20260701_001446_Th0.7_D0.3_R2 1.5と0.3/detailed_trade_log_v5_M2.csv',
 'R2_1と0.3   ':'/workspace/data/XAUUSD/stratum_7_models/backtest_simulator_results/M2_20260629_200219_Th0.7_D0.3_R2 1と0.3/detailed_trade_log_v5_M2.csv',
 'R2_X+2TF    ':'/workspace/data/XAUUSD/stratum_7_models/backtest_simulator_results/M2_20260618_105217_Th0.7_D0.3_R2 X+2TF/detailed_trade_log_v5_M2.csv',
}
for tag,path in paths.items():
    cnt={b[0]:0 for b in bins}; wins={b[0]:0 for b in bins}; tot=0; pnl=0.0
    try:
        with open(path, newline='', encoding='utf-8') as f:
            r=csv.DictReader(f)
            for row in r:
                try: td=float(row['TD'])
                except: continue
                try: lb=int(float(row['label']))
                except: lb=None
                try: pnl+=float(row['pnl'])
                except: pass
                tot+=1
                for name,lo,hi in bins:
                    if lo<td<=hi:
                        cnt[name]+=1
                        if lb==1: wins[name]+=1
                        break
    except FileNotFoundError:
        print('='*50); print(f'{tag} ← ファイル無し: {path}'); continue
    print('='*50)
    print(f'{tag} 総トレード={tot}  総pnl={pnl:,.0f}')
    print(f'  {\"TD帯\":<7}{\"件数\":>8}{\"割合%\":>9}{\"勝率%\":>9}')
    for name,_,_ in bins:
        c=cnt[name]; p=100*c/tot if tot else 0; w=100*wins[name]/c if c else 0
        print(f'  {name:<7}{c:>8}{p:>8.1f}%{w:>8.1f}%')
"