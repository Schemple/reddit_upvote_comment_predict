from datetime import datetime
from urllib.parse import urlparse
import pandas as pd
from utility import load_model


"""The function get_weekday will receive the datetime argument(Reddit posting date) and return the current weekday"""
def get_weekday(dt: datetime):
    # Get the current day of the week (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
    current_day_index = dt.weekday()

    # Define the days of the week and initialize the dictionary
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    result = {f"is{day}": 0 for day in days_of_week}

    # Set the current day to 1
    result[f"is{days_of_week[current_day_index]}"] = 1

    return result


"""Function to return the whether the Reddit posting time hour(morning, afternoon,...)"""
def categorize_time_of_day(dt: datetime):
    hour = dt.hour

    if 0 <= hour < 6:
        return {"isNight": 1, "isMorning": 0, "isAfternoon": 0, "isEvening": 0}
    elif 6 <= hour < 12:
        return {"isNight": 0, "isMorning": 1, "isAfternoon": 0, "isEvening": 0}
    elif 12 <= hour < 18:
        return {"isNight": 0, "isMorning": 0, "isAfternoon": 1, "isEvening": 0}
    else:  # from 18:00 to less than 24:00
        return {"isNight": 0, "isMorning": 0, "isAfternoon": 0, "isEvening": 1}


"""Getting the length of the string. Later we will use the post title and post content given by the user on the Website to feed into this function"""
def string_length(input_string):
    return len(input_string)


"""Function to receive the URL referenced link given by the user on the Website, 
and then return the name of the website of the URL link referenced in the Reddit post"""
def reference_external_link(url_link="", list_of_popular_webs=['Other', 'apnews', 'bbc', 'businessinsider',
       'clips', 'cnn', 'en', 'imgur', 'independent', 'mil', 'nbcnews',
       'newsweek', 'noURL', 'rawstory', 'reuters', 'streamable',
       'theguardian', 'thehill', 'twitter', 'x', 'youtube']):
    # Initialize the result dictionary with all websites in the list
    result = {website: 0 for website in list_of_popular_webs}

    # If URL link is None then return the result that there is no attached URL
    if url_link == "":
        result["noURL"] = 1
        return result

    # Parse the URL to extract the domain name
    parsed_url = urlparse(url_link)
    domain = parsed_url.netloc.split('.')[1]  # Extract the first part of the domain

    # Check if the domain matches any website in the list
    if domain in list_of_popular_webs:
        result[domain] = 1
    else:
        result["Other"] = 1

    return result


"""Counting the number of occurences for each word in a list of words in Reddit post title and post content and return a dictionary."""
def count_word_occurrences(long_string, word_list = ['able',
       'actually', 'ago', 'almost', 'already', 'also', 'always', 'amp',
       'another', 'anyone', 'anything', 'around', 'asked', 'away', 'back',
       'bad', 'believe', 'best', 'better', 'big', 'bit', 'ca', 'call',
       'called', 'came', 'car', 'care', 'come', 'comments', 'could',
       'dad', 'daughter', 'day', 'days', 'de', 'done', 'e', 'else', 'end',
       'enough', 'even', 'ever', 'every', 'everyone', 'everything',
       'family', 'feel', 'felt', 'find', 'first', 'found', 'friends',
       'fuck', 'full', 'game', 'games', 'get', 'getting', 'give', 'go',
       'going', 'good', 'got', 'great', 'gt', 'guy', 'guys', 'hard',
       'help', 'home', 'hours', 'house', 'https', 'husband', 'instead',
       'job', 'keep', 'kids', 'kind', 'know', 'last', 'later', 'league',
       'least', 'left', 'let', 'life', 'like', 'little', 'live', 'long',
       'look', 'lot', 'love', 'made', 'make', 'makes', 'making', 'man',
       'many', 'may', 'maybe', 'might', 'minutes', 'mom', 'money',
       'months', 'mother', 'much', 'na', 'name', 'need', 'never', 'new',
       'next', 'night', 'nothing', 'old', 'one', 'parents', 'part', 'pay',
       'people', 'person', 'place', 'play', 'please', 'point', 'post',
       'pretty', 'probably', 'put', 'real', 'really', 'reason', 'right',
       'said', 'saw', 'say', 'saying', 'says', 'see', 'seems', 'seen',
       'shit', 'show', 'since', 'sister', 'someone', 'something', 'start',
       'started', 'still', 'stop', 'sure', 'take', 'talk', 'talking',
       'team', 'tell', 'thing', 'things', 'think', 'though', 'thought',
       'time', 'times', 'today', 'together', 'told', 'took', 'top',
       'tried', 'trump', 'try', 'trying', 'two', 'understand', 'us',
       'use', 'used', 'video', 'want', 'wanted', 'way', 'week', 'well',
       'went', 'whole', 'wife', 'without', 'woman', 'women', 'work',
       'working', 'world', 'would', 'wrong', 'year', 'years']):
    # Split the long string into words and convert into lowercase
    words = long_string.split()
    words = [word.lower() for word in words]

    # Initialize the result dictionary with all words in the list
    result = {word: 0 for word in word_list}

    # Count the occurrences of each word in the long string
    for word in words:
        if word in word_list:
            result[word] += 1

    return result


"""Function to return a large Python dictionary indicating all informations related to 1 Reddit post. 
Through all the intermediate processing part, the informations given initially would be processed and concatenated together 
into a large Python dictionary ready to be given to the already trained Scikit-learn Machine Learning model afterward.
We will use all the above functions to process the data."""
def inference_items(subscribers, posting_time: datetime, posting_title, posting_content,
                    referenced_url=""):
    """
    Predicting target given all the other informations

    Parameters:
    subscribers: Number of subscribers for this Reddit user (int)
    posting time: the time when the Reddit post appear (datetime)
    posting title: the title of the Reddit post (string)
    posting content: the content of the Reddit post (string)
    referenced_url: the URL link referenced in the post(can be None)

    return: the large Python dictionary combining all the above informations.
    """

    def sum_dictionaries(dict1, dict2):
        result = {}

        # Iterate over the keys of the dictionaries
        for key in dict1.keys() | dict2.keys():
            # Add the corresponding values if the key exists in both dictionaries
            result[key] = dict1.get(key, 0) + dict2.get(key, 0)

        return result

    subscriber_dict = {"subscribers": subscribers}

    weekday_dict = get_weekday(posting_time)

    posting_hour_dict = categorize_time_of_day(posting_time)

    title_length_dict = {"title_length": string_length(posting_title)}

    selftext_length_dict = {"selftext_length": string_length(posting_content)}

    referenced_url_dict = reference_external_link(referenced_url)

    title_word_count_dict = count_word_occurrences(posting_title)

    content_word_count_dict = count_word_occurrences(posting_content)

    concat_word_count_dict = sum_dictionaries(title_word_count_dict, content_word_count_dict)

    list_dict = [subscriber_dict, weekday_dict, posting_hour_dict, title_length_dict,
                 selftext_length_dict, referenced_url_dict, concat_word_count_dict]
    final_dict = {}
    for dictionary in list_dict:
        final_dict.update(dictionary)

    return final_dict


"""API function to predict the number of voteups for 1 Reddit post. The Extra Tree model is the best model and would be used here"""
def predict_voteups_api(subscribers, posting_time: datetime, posting_title, posting_content,
                    referenced_url=""):

    estimator = load_model("models/ExtraTree_voteups.pkl")

    final_dict = inference_items(subscribers, posting_time, posting_title, posting_content, referenced_url)
    transformed_item = pd.DataFrame(final_dict, index=[0])[estimator.feature_names_in_]
    predicted_value = estimator.predict(transformed_item)[0]

    return predicted_value


"""API function to predict the number of comments for 1 Reddit post. The Extra Tree model is the best model and would be used here"""
def predict_comments_api(subscribers, posting_time: datetime, posting_title, posting_content,
                        referenced_url=""):
    estimator = load_model("models/ExtraTree_comments.pkl")

    final_dict = inference_items(subscribers, posting_time, posting_title, posting_content, referenced_url)
    transformed_item = pd.DataFrame(final_dict, index=[0])[estimator.feature_names_in_]
    predicted_value = estimator.predict(transformed_item)[0]

    return predicted_value