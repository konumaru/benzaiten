# Todo

- [ ] transformerモデルの実装
- [ ] 学習データの追加
- [ ] pytorch lightning導入
- [ ] 出力ファイルの余計な空白の除去
- [ ] モデルのバージョン管理


## 学習データ

- 新しく配布されたものを追加する
  - confingをA,Minorに対応させる必要がある
  - major, minorメロディー生成するモデルを分けるべきかもしれない


## 出力について

- `generate.py`を実行
- あらかじめ用意してあった複数のモデルの複数回実行して、メロディを生成
- 生成結果を比較して選びたい
- コンペ特有の制約
  - 当日サンプル同様のmidi fileが配布される
  - pathを指定したらそれに切り替わるようにしたい
- output tree
  - output/
    - {input_backing_filename}  # e.g. sample1_backing
      - model_name/  # e.g. LSTM, Transformer, LSTM_short_seq, Transrformer_small_batch
        - output.midi
        - output.wav
        - piano_roll.png
        - config.yaml
      - play_space/
        - model_name.wav

## モデル

- ハンズオンのLSTM
- Music Transformer
- (Magenta)
  - Tensorflow製なので実装膨らみそうだから避けたい
  - 一方で学習済みモデルなので使いたい気持ちもある
