import jpx_data
import pandas as pd

# JPXに上場している全銘柄のリストを取得
df = jpx_data.get_stock_list_info()

# 銘柄コードと会社名のみを抽出
stock_list = df[['Local Code', 'CompanyName']].rename(columns={'Local Code': 'Code', 'CompanyName': 'Name'})

# CSVファイルとして保存
stock_list.to_csv("stock_list.csv", index=False, encoding='utf-8-sig')

print("全銘柄リストが'stock_list.csv'として保存されました。")