
from googleapiclient.discovery import build
import pandas as pd 
import numpy as np
import seaborn as sns
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import gensim
import re


# api key for executing is
api = "AIzaSyD4VNNviHG59Dng3qXbBvX-vMjjqArS5mc"

import requests

def get_video_ids(api,search_query,number_of_results):

    URL = "https://www.googleapis.com/youtube/v3/search"
    PARAMS = {
        "key": api,
        "q": search_query,
        "part": "snippet",
        "maxResults": number_of_results,
        "type": "video",
    }
    response = requests.get(URL, params=PARAMS)
    if response.status_code == 200:
        #print("========> Successful request with status code 200 ok")
        videos = response.json()["items"]
        #print("========> Number of videos retrieved is = ",len(videos))
        video_ids = []
        for i in range(len(videos)):
            id = videos[i]["id"]['videoId']
            video_ids.append(id)
        return video_ids
    else:
        pass
        #print("========> Failed request with status code 400")

def get_all_comments(api,video_id,MinLikesForComments):   
    youtube = build('youtube','v3',developerKey=api)
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    comments = {"Comments":[],
                "Likes Count":[],
                "Tag":[],
                "Sentiment":[],
                "Conf Val":[]}
    
    params = {
        "part": "snippet",
        "videoId": "jtn-hRJjl68",
        "key": api,
        "maxResults": 100,
        "order": 'relevance'
    }
    request = youtube.commentThreads().list(videoId=video_id,part="id,snippet,replies",maxResults=100)
   
    while request:
        response = request.execute()
        for comment in response['items']:
            if comment['snippet']['topLevelComment']['snippet']['likeCount'] > MinLikesForComments:
                #print(comment['snippet']['topLevelComment']['snippet']['likeCount'])
                com_text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
                sentiment,conf_val = get_sentiment(com_text)
                comments['Comments'].append(com_text)
                comments['Likes Count'].append(comment['snippet']['topLevelComment']['snippet']['likeCount'])
                comments['Tag'].append("comment")
                comments['Sentiment'].append(sentiment)
                comments['Conf Val'].append(conf_val)
                #print(len(comments['Comments']))
                rc = comment['snippet']['totalReplyCount']
                reply_lt = comment.get('replies')
                if reply_lt is not None and rc != len(reply_lt['comments']):
                    replies = get_comment_replies(youtube,comment['id'],MinLikesForComments)
                    comments["Comments"].extend(replies['Comments'])
                    comments["Likes Count"].extend(replies['Likes Count'])
                    comments["Tag"].extend(replies['Tag'])
                    comments['Sentiment'].extend(replies['Sentiment'])
                    comments['Conf Val'].extend(replies['Conf Val'])
            else:
                comments = {"Comments":[],
                    "Likes Count":[],
                    "Tag":[],
                    "Sentiment":[],
                    "Conf Val":[]}

    
        df = pd.DataFrame(comments)
        request = youtube.commentThreads().list_next(request, response)
    return df

def get_comment_replies(youtube, comment_id,MinLikesForComments):
    request = youtube.comments().list(
        parentId = comment_id,
        part = 'id,snippet',
        maxResults = 100
    )
    replies = {"Comments":[],
               "Likes Count":[],
               "Tag":[],
               "Sentiment":[],
               "Conf Val":[]}
    while request:
        response = request.execute()
        
        reply_items = response['items']
        for i in reply_items:
            if i['snippet']['likeCount'] > MinLikesForComments:
                reply_text = i['snippet']['textDisplay']
                sentiment,conf_val = get_sentiment(reply_text)
                replies["Comments"].append(reply_text)
                replies["Likes Count"].append(i['snippet']['likeCount'])
                replies["Tag"].append("Reply")
                replies['Sentiment'].append(sentiment)
                replies['Conf Val'].append(conf_val)
        request = youtube.comments().list_next(request, response)
    return replies

def calculate_coherence(title, description, comments):
    # Remove links and time stamps from the description
    description = re.sub(r'http\S+', '', description)
    description = re.sub(r'\d+:\d+', '', description)
    text_data = [title, description] + comments

    # Preprocessing the text data
    text_data = [sentence.split() for sentence in text_data]

    # Training the Word2Vec model
    model = gensim.models.Word2Vec(text_data, window=5, min_count=1, workers=4)

    # Generating word embeddings
    word_vectors = model.wv

    # Defining a function to calculate cosine similarity between two vectors
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # Calculating cosine similarity between title and description
    title_vec = np.zeros(100)
    for word in title.split():
        title_vec += word_vectors[word]
    description_vec = np.zeros(100)
    for word in description.split():
        description_vec += word_vectors[word]
    title_description_similarity = cosine_similarity(title_vec, description_vec)

    # Calculating cosine similarity between title, description, and comments
    comments_vec = np.zeros(100)
    for comment in comments:
        for word in comment.split():
            comments_vec += word_vectors[word]
    title_description_comments_similarity = cosine_similarity(title_vec + description_vec, comments_vec)

    return title_description_similarity, title_description_comments_similarity


def get_sentiment(text):
    vader = SentimentIntensityAnalyzer()
    sentiment = vader.polarity_scores(text)
    max_key = max(sentiment, key=lambda k: sentiment.get(k))
    sentiment_val = sentiment[max_key]
    if sentiment_val > 0.4:
        return "pos", sentiment_val
    elif sentiment_val < -0.4:
        return "neg", sentiment_val
    else:
        return None


def get_overall_sentiments(df,comments_df):
    data = df.copy()
    conf_vals = []
    sentiments = []
    #if len(comments_df) != 0:
    for i in data.iterrows():
        #print("type of comments_df: ",type(i[1]['Comments Data']),i[1]['Comments Data'])
        if len(comments_df) > 0: #len(i[1]['Comments Data']) > 0:
            # getting comments with atleast 3 likes...
            #             print("type of comments_df: ",type(i[1]['Comments Data']),i[1]['Comments Data'])
            dv = i[1]['Comments Data'].copy()
            dv = dv[dv['Likes Count']>=3]
            sentiments.append(dv['Sentiment'].mode().loc[0])
            conf_vals.append(dv['Conf Val'].mean())
        else:
            sentiments.append("")
            conf_vals.append(0)

    data['Sentiemnt'] = sentiments
    data['Conf Val'] = conf_vals
    return data

def driver(api,search_query,number_of_videos,MinLikesForComments = 50):
    youtube = build('youtube','v3',developerKey=api)
    video_ids = get_video_ids(api,search_query,number_of_videos)
    # loop through video ids and get the comments...
    rows = []
    for i in video_ids:
        final_result = {}
        video_data = youtube.videos().list(part='snippet,statistics', id=i).execute()
        channel_id = video_data['items'][0]['snippet']['channelId']
        channel_data = youtube.channels().list(part='snippet,contentDetails,statistics', id=channel_id).execute()
        final_result['Channel ID'] = channel_id
        final_result['Query'] = search_query
        final_result['Video ID'] = i
        final_result['Channel Name'] = channel_data['items'][0]['snippet']['title']
        final_result['Channel Description'] = channel_data['items'][0]['snippet']['description']
        final_result['Subscribers Count'] = channel_data['items'][0]['statistics']['subscriberCount']
        final_result['Video Count'] = channel_data['items'][0]['statistics']['videoCount']
        final_result['Channel View Count'] = channel_data['items'][0]['statistics']['viewCount']
        final_result['Title'] = video_data['items'][0]['snippet']['title']
        final_result['Description'] = video_data['items'][0]['snippet']['description']
        final_result['Uploaded Time'] = video_data['items'][0]['snippet']['publishedAt']
        final_result['View Count'] = video_data['items'][0]['statistics']['viewCount']
        final_result['Like Count'] = video_data['items'][0]['statistics']['likeCount']
        item = video_data['items'][0]
        #if 'snippet' in item: #and 'commentsDisabled' in item['snippet']: #and item['snippet']['commentsDisabled']:
        if 'commentCount' in video_data['items'][0]['statistics'] and int(item['statistics']['commentCount']) >0:
            #print(video_data['items'][0]['statistics']['commentCount'])
            comments_df = get_all_comments(api,i,MinLikesForComments)
            final_result['Comments Data'] = comments_df
            title = video_data['items'][0]['snippet']['title']
            description = video_data['items'][0]['snippet']['description']
            comments = comments_df['Comments'].to_list()
            coherence, cohesion = calculate_coherence(title, description, comments)
            final_result['Comment Count'] = video_data['items'][0]['statistics']['commentCount']
            final_result['Coherence_Score'] = coherence
            final_result['Cohesion_Score'] = cohesion
        else:
            comments_df = pd.DataFrame({"Comments":[],"Likes Count":[],"Tag":[],"Sentiment":[],"Conf Val":[]})
            #print(len(comments_df))
            final_result['Coherence_Score'] = 0
            final_result['Cohesion_Score'] = 0
            final_result['Sentiemnt'] = "No Sentiment"
            final_result['Conf Val'] = 0
        rows.append(final_result)
    df = pd.DataFrame(rows)
    #if int(item['statistics']['commentCount']) >0:
    # getting overall sentiments of each video..
    df = get_overall_sentiments(df,comments_df)
    return df