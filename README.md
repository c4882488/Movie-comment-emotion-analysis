# [Movie-comment-emotion-analysis](https://github.com/c4882488/Movie-comment-emotion-analysis)

影評情感分析

-------- 

## 介紹

透過NLP實現影評的情感分析，可以讓片商們了解到現今觀眾普遍對於該片的想法，有什麼是符合大眾的口味。

### Data Source

來自於Yahoo電影 (哥吉拉大戰金剛、鬼滅之刃劇場版、玩命鈔劫、尋龍使者、惡水真相、父親、那些要我死的人、聽見歌 再唱、死亡漩渦：奪魂鋸新遊戲、當男人戀愛時)

+ Features : 影評文字

+ Label :影評大於4顆星為推薦影片，影評小於4顆星為不推薦影片

### 方法

Ckiptagger ▷ Tfidf ▷ Stop-word ▷ Logistic regression ▷ GridSearchCV ▷ 

### 績效表





| ![Picture1.png](https://raw.githubusercontent.com/c4882488/c4882488/main/2023/02/07-15-23-11-Picture1.png)                                                                                                          | ![Picture2.png](https://raw.githubusercontent.com/c4882488/c4882488/main/2023/02/07-15-23-21-Picture2.png)                                                                                                          |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 模型準確率                                                                                                                                                                                                               | ROC AUC圖                                                                                                                                                                                                            |
| <img src="https://raw.githubusercontent.com/c4882488/c4882488/main/2023/02/07-15-08-22-CleanShot%202023-02-07%20at%2015.08.13%402x.jpg" title="" alt="CleanShot 2023-02-07 at 15.08.13@2x.jpg" data-align="center"> | <img src="https://raw.githubusercontent.com/c4882488/c4882488/main/2023/02/07-15-08-45-CleanShot%202023-02-07%20at%2015.08.36%402x.jpg" title="" alt="CleanShot 2023-02-07 at 15.08.36@2x.jpg" data-align="center"> |
| 混淆矩陣                                                                                                                                                                                                                | 衡量指標                                                                                                                                                                                                                |

### 限制

+ 資料筆數少

+ 評價好得多過於壞的

+ Ckiptagger 斷詞有些不適當

---

詳細的程式說明
