# 使用方法
## 各ディレクトリ・ファイルについて
- data ROHANの動画のdataおよびannotation dataを格納しておく（これらについてはhttps://zunko.jp/multimodal_dev/twdashbord.php参照）
- pretrain 元のLIPNETの各パラメータの重み
- scripts 唇抽出用
- annotation.py　次章参照
- cvtransforms.py データの水増し、加工
- dataset.py　ファイル読み込みやcerの計算の定義等
- model.py ネットワークの構造を定義（LIPNETをベースにした構造）
- option.py 各種ハイパーパラメータの調整
- main.py trainとtest、lossの計算
 - main.py, main3.py　特にペナルティなし
 - main2.py silとpauにpenalty
 - main4.py 頻度の逆比にpenalty
 - main5.py　silとpauと母音にpenalty
 - main6.py silとpauを損失計算から除去

## データセットの作り方
data直下に
①annotation
②extracted_lip
③original_mov
④original_mov_before
の4フォルダを作成。

④に動画のmp4ファイルをフォルダごと，①に.labファイルをぶちこむ

data内の`select_mp4.py`でmp4ファイルだけ取り出し③に格納（元々mp4ファイルだけしかなければここにぶち込んでok）
scripts内の `extract_lip.py`を実行すると，②に唇領域のみ切り出した画像が生成

`create_data_lists.py`を実行すると，訓練データとテストデータをランダムに分け，main.pyで指定するためのテキストファイルが生成される

`annotaion.py`を実行するとdata直下のannotaion内にある.labファイルについてフレームごとに音素を割り当てたものをdata直下processedに生成

パスとかはよしなに変更してください

## 実装方針
- 元々のLIP-NET-JPだとpredictの文字数が異常に長くなるという問題点があった
→ とりあえず元の.labファイルをフレーム数で分割してそのフレームに対応する音素を.labファイルに書かれた各音素の開始時間、終了時間を元に割り振り
→ これによってpredictとtruthが１対１対応に
→　いったんsilとpauも含めて全て出力出力、同じ音素の繰り返しも全て出力という形の実装