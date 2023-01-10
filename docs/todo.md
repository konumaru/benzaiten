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

## 入力データへの疑問

- 特定の音が何小節鳴り続けているかを考慮できてない？
  - music xml の入力を単純に List に Append しているだけに見える
    - ただし休符は扱えてる？
