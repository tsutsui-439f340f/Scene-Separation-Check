# Scene-Separation-Check
自動シーン分離における課題の一つに、分離結果の人手確認のコスト問題があります。
その問題を改善できるように、入力されたシーンデータが1つになっているかを判定する2値分類システムを作成しました。\
システムの学習には独自で収集しているアニメ調の動画の3800シーンを利用。
学習時にはシーンをランダムに組み合わせ、ランダムに60フレームを取得することで教師データを作成しています。


![image](https://user-images.githubusercontent.com/55880071/185200244-e66a1d71-cbe9-4650-bd84-789e85bd7012.png)
<div align="center">

<img src="https://user-images.githubusercontent.com/55880071/185285516-363095ee-2d89-4c97-b271-1ae72ff58ec5.png" width=400>
<img src="https://user-images.githubusercontent.com/55880071/185285475-5a39469d-0bda-4681-a42e-eed240ff24ce.png" width=400>
</div>
<div align="center">
![image](https://user-images.githubusercontent.com/55880071/185201345-850ff375-dcdd-423d-8600-4948030ed13d.png)
</div>
