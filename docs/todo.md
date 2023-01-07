# Todo

- [ ] 学習データの追加
- [ ] 前処理を理解するためにテストを書く
- [ ] Many to Many の LSTM モデルで学習
- [ ] music transformer モデルの実装

## 学習データ

- 新しく配布されたものを追加する

  - confing を A,Minor に対応させる必要がある
  - major, minor メロディー生成するモデルを分けるべきかもしれない

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

## 生成

以下のように実行したら生成されてほしい

```sh
python generate.py \
  demo.name=sample1 \
  exp.name=LSTM,LSTM_batchSize-2,LSTM_hiddenDim-256
```
