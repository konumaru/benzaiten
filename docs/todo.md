# Todo

- [ ] 前処理を理解するためにテストを書く
  - [ ] 生成時に入力してる note_seq の中身がすべて 0 なのかを確認する（backing.midi をつかってる？）
- [ ] Many to Many の LSTM モデルで学習
- [ ] music transformer モデルの実装, ここまでやりたい

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

### モデル改善のアイディア

- 扱う seq_len を長くする
- 学習時にメロディのデータを入力に使用しない
  - backing.midi はあるからそれは使ってもいいのか
- 双方向 LSTM にしてみる
