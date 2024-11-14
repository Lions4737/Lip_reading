# 使用方法
## データセットの作り方
data直下に
①annotation
②extracted_lip
③original_mov
の3フォルダを作成。

③に動画のmp4ファイルをフォルダごと，①に.labファイルをぶちこむ

scripts内の `extract_lip.py`を実行すると，②に唇領域のみ切り出した画像が生成

`create_data_lists.py`を実行すると，訓練データとテストデータをランダムに分け，main.pyで指定するためのテキストファイルが生成される