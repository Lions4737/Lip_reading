# 使用方法
## データセットの作り方
data直下に
①annotation
②extracted_lip
③original_mov
④original_mov_before
の4フォルダを作成。

④に動画のmp4ファイルをフォルダごと，①に.labファイルをぶちこむ

data内のselect_mp4.pyでmp4ファイルだけ取り出し③に格納（元々mp4ファイルだけしかなければここにぶち込んでok）
scripts内の `extract_lip.py`を実行すると，②に唇領域のみ切り出した画像が生成

`create_data_lists.py`を実行すると，訓練データとテストデータをランダムに分け，main.pyで指定するためのテキストファイルが生成される

`annotaion.py`を実行するとdata直下のannotaion内にある.labファイルについてフレームごとに音素を割り当てたものをdata直下processedに生成

パスとかはよしなに変更してください

## 実装方針
- 元々のLIP-NET-JPだとpredictの文字数が異常に長くなるという問題点があった
→ とりあえず元の.labファイルをフレーム数で分割してそのフレームに対応する音素を.labファイルに書かれた各音素の開始時間、終了時間を元に割り振り
→ これによってpredictとtruthが１対１対応に
→　いったんsilとpauも含めて全て出力出力、同じ音素の繰り返しも全て出力という形の実装