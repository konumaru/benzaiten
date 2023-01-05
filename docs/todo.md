# Todo

- [ ] 異なる学習パラメータでモデルを学習
  - [ ] batch_size=2,4,16
  - [ ] seq_len=256,512,2924
- [ ] 前処理を理解するためにテストを書く
- [ ] 学習データの追加
- [ ] pytorch lightning 導入
- [ ] Many to Many の LSTM モデルで学習
- [ ] music transformer モデルの実装

## 学習データ

- 新しく配布されたものを追加する
  - confing を A,Minor に対応させる必要がある
  - major, minor メロディー生成するモデルを分けるべきかもしれない

## 出力について

- `generate.py`を実行
- あらかじめ用意してあった複数のモデルの複数回実行して、メロディを生成
- 生成結果を比較して選びたい
- コンペ特有の制約
  - 当日サンプル同様の midi file が配布される
  - path を指定したらそれに切り替わるようにしたい
- output tree
  - generated/
    - {input_backing_filename} # e.g. sample1_backing
      - exp_name/ # e.g. LSTM, LSTM_batch_size=2,seq_len=1024,
        MusicTransformer
        - output.midi
        - output.wav
        - piano_roll.png
        - config.yaml
      - play_space/
        - model_name.wav

## モデル

- ハンズオンの LSTM
  - Many to one
  - Many to many
  - 双方向 LSTM
- Music Transformer
- (Magenta)
  - Tensorflow 製なので実装膨らみそうだから避けたい
  - 一方で学習済みモデルなので使いたい気持ちもある

## その他メモ

- batch_size は小さく、seq_len を長く、がいいっぽい？
