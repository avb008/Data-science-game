# Data-science-game
It is an International Data Science competition for Students world wide where top 20 teams in prelims qualify to the finals in Paris every year.

Prelims Objective : 
Deezer is a music streaming app which has its own music recommendation system based on collaborative filtering algorithm on its users . The main goal is to Devise a solution which can predict whether a user listens to the first track recommended by the deezer app.

Approach:
Our idea is to learn embeddings of users,songs,genres based on the interaction of features and output which inturn are used for prediction purpose and then use these embeddings along with other features to create the Classfication model.
Embeddings were created for User_id ,genre_id,artist_id,media_id
Users : (Age,gender ,user_id)
Genres : (genre_id)
Songs : (media_id)
Artists : (Artist_id)

Then We used Emeddings and other features as inputs to XGBoost to predict whether a user listens to the first track recommended by the deezer app.
