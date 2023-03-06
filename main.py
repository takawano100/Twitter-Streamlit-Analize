import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image


st.title('プログラミング教室 Twitter 分析結果')

"""
# これまでの作業の説明

１．３月４日
  (1)Twitter_APIを使って「プログラミング教育」「プログラミング教室　小学生」で
  つぶやきを検索
  - 「プログラミング教育」 → 956件（過去１周間）
  - 「プログラミング教室　小学生」 → 107件（過去１周間）

  (2)発言の多い人を週出
  - 「プログラミング教育」 → 16人
  - 「プログラミング教室　小学生」 → 7人

  (3)発言者の内、フォロワーが多い（1,000人以上）に絞る
  7人がフォロワー 1,000 人以上

  (4)7人をフォローをしているユーザーを抽出
 
   - 98507061 : 2,018 人
   - 1100479543 : 10,000 人(上限)
   - 1595290706 : 1,923 人
   - 4485056533 : 2,837 人
   - 1083565539933315072 : 8,663 人
   - 1086130739932844033 : 2,960 人
   - 1230342829706137601 : 3,558 人
   - 合計  31,959 人 フォロワー（重付呪があるかも）

２．３月５日
 (1)上記の中から、name または description に「ママ」がある人を抽出
   - 98507061 : 33 人
   - 1100479543 : 175 人
   - 1595290706 : 6 人
   - 4485056533 : 4 人
   - 1083565539933315072 : 2,061 人
   - 1086130739932844033 : 21 人
   - 1230342829706137601 : 246 人
   - 合計  2,546 人

 (2) 上記の「ららこ＠子供プログラミング教室口コミいっぱい書きました。(1083565539933315072)」
    location と description に「大阪」を含む人を抽出 38人
    https://xn--9ckk2d5c4051a8fm.xyz/

 (3)Twitter API の制限でつぶやきを 14人分抽出した。
  - プロフィール : name, username, 写真, url(あれば)
  - 日々のつぶやき
  - 自分のブログやサイトを持っていて、リテラシーは有りそう
"""
"""
# Twitter 一覧表の見かた
- id : 内部コード（通常は見えない）
- name : 名前
- username : @に続くID
- created_at : 作成年月
- protected : ？
- whitheld : ？
- location : 場所（任意）
- url : リンク
- description : プロフィール（事故紹介）
- verified : ？
- entities : ？
- profile_images_url : 画像
- public_metrics : 各種指標
-  followers_count : フォロワー数
-  folling_count : フォロー数
-  tweet_count : tweet数
-  listed_count : ？
- pinned_tweetd_id : 固定したID

"""
"""
# 抽出したフォロワーさん一覧
"""

df_1 = pd.read_csv('./csv/twitter_user_info_1.csv')
df_2 = pd.read_csv('./csv/twitter_user_info_2.csv')

st.text('「プログラミング教室　小学生」 : 9 人')
st.dataframe(df_1)
st.text('「プログラミング教育」 : 16 人')
st.dataframe(df_2)

"""
# 抽出したママさんの一覧
"""
df1 = pd.read_csv('./csv/df1_new.csv')
df2 = pd.read_csv('./csv/df2_new.csv')
df3 = pd.read_csv('./csv/df3_new.csv')
df4 = pd.read_csv('./csv/df4_new.csv')
df5 = pd.read_csv('./csv/df5_new.csv')
df6 = pd.read_csv('./csv/df6_new.csv')
df7 = pd.read_csv('./csv/df7_new.csv')

st.text('鎌田裕二『何でも相談できるパソコン駆込み寺』（横浜市鶴見区）(98507061) : 33 人')
st.dataframe(df1)
st.text('ICT教育ニュース(1100479543) : 175 人')
st.dataframe(df2)
st.text('名古屋文理大学 情報メディア学科(1595290706) : 6 人')
st.dataframe(df3)
st.text('nest＋＋(4485056533) : 4 人')
st.dataframe(df4)
st.text('ららこ＠子供プログラミング教室口コミいっぱい書きました。(1083565539933315072) : 2,061 人')
st.dataframe(df5)
st.text('現役エンジニアが選ぶ@プログラミングスクール比較bot(1086130739932844033) : 21 人')
st.dataframe(df6)
st.text('キャリアハブ(1230342829706137601) : 246 人')
st.dataframe(df7)

"""
# 大阪のママさんの一覧
"""
df5_osaka = pd.read_csv('./csv/df5_osaka.csv')


st.text('ららこ＠子供プログラミング教室口コミいっぱい書きました。(1083565539933315072) : 38 人')
st.dataframe(df5_osaka)

"""
# 大阪のママ達のつぶやき（一部）
"""
termas_jp = pd.read_csv('./person/@termas_jp.csv')
I_termas_jp = Image.open('./images/termas_jp.jpg')
st.text('てるま(termas_jp)さん : 266 件')
st.image(I_termas_jp, caption='termas_jp')
"""
https://termas.jp/
"""
st.dataframe(termas_jp)

kidstraveler22 = pd.read_csv('./person/@kidstraveler22.csv')
I_kidstraveler22 = Image.open('./images/kidstraveler22.jpg')
st.text('ハルカ＠子マル家３歳・１歳娘と子連れ旅行(kidstraveler22)さん : 146 件')
st.image(I_kidstraveler22, caption='kidstraveler22')
"""
https://kids-traveler.com/profile
"""
st.dataframe(kidstraveler22)

sakuraich21 = pd.read_csv('./person/@sakuraich21.csv')
I_sakuraich21 = Image.open('./images/sakuraich21.jpg')
st.text('さくらいch@親子で楽しく遊ぶVLOG(@sakuraich21)さん : 148 件')
st.image(I_sakuraich21, caption='I_sakuraich21')
"""
https://www.youtube.com/channel/UCdBcANF7bdWUlu7nRL4ukIw
"""
st.dataframe(sakuraich21)

keikoyokoe = pd.read_csv('./person/@keikoyokoe.csv')
I_keikoyokoe = Image.open('./images/keikoyokoe.jpg')
st.text('けいこ@ママの心と身体を整えるお手伝い♡(@keikoyokoe)さん : 2,186 件')
st.image(I_keikoyokoe, caption='keikoyokoe')
st.dataframe(keikoyokoe)

XpXHBZH9OEEuTwd = pd.read_csv('./person/@XpXHBZH9OEEuTwd.csv')
I_XpXHBZH9OEEuTwd = Image.open('./images/XpXHBZH9OEEuTwd.jpg')
st.text('みずくらげちゃん(@XpXHBZH9OEEuTwd)さん : 94 件')
st.image(I_XpXHBZH9OEEuTwd, caption='XpXHBZH9OEEuTwd')
st.dataframe(XpXHBZH9OEEuTwd)

hideyuki_k1 = pd.read_csv('./person/@hideyuki_k1.csv')
I_hideyuki_k1 = Image.open('./images/hideyuki_k1.jpg')
st.text('ひで😄子育て応援＆すべての人に笑顔を！(@hideyuki_k1)さん : 644 件')
st.image(I_hideyuki_k1, caption='hideyuki_k1')
st.dataframe(hideyuki_k1)

yukarin = pd.read_csv('./person/@257yukarin.csv')
I_yukarin = Image.open('./images/yukarin.jpg')
st.text('ゆかりん🌸(@257yukarin)さん : 3,000 件(最大)')
st.image(I_yukarin, caption='257yukarin')
st.dataframe(yukarin)

ito_egao = pd.read_csv('./person/@ito_egao.csv')
I_ito_egao = Image.open('./images/ito_egao.jpg')
st.text('いと(@ito_egao)さん : 3,000 件(最大)')
st.image(I_ito_egao, caption='ito_egao')
"""
https://lit.link/itoo
"""
st.dataframe(ito_egao)

WokingmamaENARI = pd.read_csv('./person/@WokingmamaENARI.csv')
I_WokingmamaENARI = Image.open('./images/WokingmamaENARI.jpg')
st.text('フルタイム勤務ワーママ🌗1歳4歳ママ👧ENARI💛(@WokingmamaENARI)さん : 455 件')
st.image(I_WokingmamaENARI, caption='WokingmamaENARI')
st.dataframe(WokingmamaENARI)

erigonmama = pd.read_csv('./person/@erigonmama.csv')
I_erigonmama = Image.open('./images/erigonmama.jpg')
st.text('歯科医師🦷えりママ👨‍👩‍👧‍👧楽天ROOM・pippin・アマアソ(@erigonmama)さん : 2,942 件')
st.image(I_erigonmama, caption='erigonmama')
"""
https://lit.link/erimama
"""
st.dataframe(erigonmama)

maris_tk = pd.read_csv('./person/@maris_tk.csv')
I_maris_tk = Image.open('./images/maris_tk.jpg')
st.text('Mariko✧SNS広報フリーランス｜プチプラコーデYouTuber(@maris_tk)さん : 1,588 件')
st.image(I_maris_tk, caption='maris_tk')
"""
https://lit.link/Marikotka
"""
st.dataframe(maris_tk)

haluulala = pd.read_csv('./person/@_haluulala_.csv')
I_haluulala = Image.open('./images/haluulala.jpg')
st.text('haluulala🌷ママグラファー@大阪(@_haluulala_)さん : 25 件')
st.image(I_haluulala, caption='haluulala')
"""
https://lit.link/Marikotka
"""
st.dataframe(haluulala)

kunpapa7 = pd.read_csv('./person/@kunpapa7.csv')
I_kunpapa7 = Image.open('./images/kunpapa7.jpg')
st.text('くんパパ (卵アレルギーっ子の父)(@kunpapa7)さん : 3,000 件(最大)')
st.image(I_kunpapa7, caption='kunpapa7')
st.dataframe(kunpapa7)

EasyMomFarmacy = pd.read_csv('./person/@EasyMomFarmacy.csv')
I_EasyMomFarmacy = Image.open('./images/EasyMomFarmacy.jpg')
st.text('ゆるママ7歳4歳☆育児/英語/お薬の色々(@EasyMomFarmacy)さん : 2,213 件')
st.image(I_EasyMomFarmacy, caption='EasyMomFarmacy')
"""
https://note.com/egg_allergy/
"""
st.dataframe(EasyMomFarmacy)

# sasapandayasu = pd.read_csv('@sasapandayasu.csv')
# st.text('ささぱんだやす(@sasapandayasu)さん : 309 件')
# st.dataframe(sasapandayasu)