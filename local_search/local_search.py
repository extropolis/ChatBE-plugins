import json
import os
import string
from typing import Any, Callable

import houndify
import requests

from ..base import BaseTool

DEFAULT_REQ_INFO = {
  "Latitude": 43.6633,
  "Longitude": -79.3957,
  "UnitPreference": "METRIC",
}

def second_to_time_string(second):
    # don't mind about the seconds, and there should be no seconds anyways
    if second > 86400:
        second = 86400 # avoid weird results into second day
    minute_from_zero = int(second / 60)
    h = int(minute_from_zero / 60)
    m = minute_from_zero % 60
    return f"{h:>02}:{m:>02}"

class LocalSearch:
    def __init__(self, user_id="test_user", req_info=DEFAULT_REQ_INFO, top_n=3):
        self.client_id = os.environ["HOUNDIFY_ID"]
        self.client_key = os.environ["HOUNDIFY_KEY"]
        self.user_id = user_id
        self.client = houndify.TextHoundClient(self.client_id, self.client_key, self.user_id, req_info)
        self.top_n = top_n

    def update_req_info(self, req_info: dict=None):
        '''
            Used when user's GPS location changed. This function will be automatically called in query_houndify
        '''
        if req_info is None:
            return
        for k, v in req_info.items():
            self.client.setHoundRequestInfo(k, v)

    def search(self, query, req_info=None):
        self.update_req_info(req_info)
        response = self.client.query(query)
        try:
            self.client.setConversationState(response["AllResults"][0]["ConversationState"])
        except Exception as e:
            print(e)
            # should change to logger later on
        # return response["AllResults"][0]['SpokenResponseLong']
        return self.retrieve_useful_data(response, query)

    def parse_native(self, local_results: list, top_n = 3, max_words=1000):
        '''
            Remove useless fields in local_results. Keep only the fields that are useful for GPT to summarize:

                {
                    "Name" (Required): name of the place,
                    "Location" (Required): only keep the address,
                    "Rating" (Optional): 0-5 rating score, sort from high to low in the list,
                    "PhoneNumber" (Optional): probably useful for restaurants,
                    "Distance" (Optional): distance from user's current location, sort from closest to farthest,
                    "Reviews" (Optional): [
                        "Rating": user rating,
                        "Text": user review,
                    ],
                    "HoursToday" (Option): when is it open today? Assume the correctness of Yelp result,
                    "Attributes" (Option): Other attributes for consideration.
                }

            Inputs:
                top_n: number of places to summarize,
                max_words: maximum number of words (an estimate of # tokens) in the json list.

            The returned list will not exceed top_n results or max_words length, whichever is shorter.
        '''
        try:
            # closer and higher rating first
            local_results = sorted(local_results, key=lambda v: (v["DistanceFromUser"]["Value"], -v["Rating"]))
        except:
            pass

        summarized_results = []
        total_words_approx = 0

        for place in local_results:
            try:
                curr_result = {
                    "Name": place["Name"],
                    "Location": place["Location"]["Address"], 
                    # if either of these 2 values DNE, we should not include the place
                }
            except:
                continue
            if "Rating" in place:
                curr_result["Rating"] = place["Rating"]
            if "Phone" in place and "Number" in place["Phone"]:
                curr_result["PhoneNumber"] = place["Phone"]["Number"]
            if "DistanceFromUser" in place:
                dist_val = place["DistanceFromUser"]["Value"]
                dist_unit = place["DistanceFromUser"]["Unit"]
                curr_result["Distance"] = f"{dist_val} {dist_unit}"
            if "Reviews" in place and len(place["Reviews"]) > 0:
                reviews = []
                # keep only the top 3 (latest) reviews
                for i in range(min(len(place["Reviews"]), 3)):
                    reviews.append({
                        "Rating": place["Reviews"][i]["Rating"],
                        "Text": place["Reviews"][i]["Text"],
                    })
                curr_result["Reviews"] = reviews
            if "HoursToday" in place:
                # hours today is given by StartTime and EndTime in seconds from 0:00
                start_time = second_to_time_string(place["HoursToday"][0]["StartTime"])
                end_time = second_to_time_string(place["HoursToday"][0]["EndTime"])
                curr_result["HoursToday"] = f"{start_time} to {end_time}"
            if "Attributes" in place:
                curr_result["Attributes"] = place["Attributes"]
            summarized_results.append(curr_result)
            
            total_words_approx += len(json.dumps(curr_result).split())

            if len(summarized_results) >= top_n or total_words_approx >= max_words:
                break
        return summarized_results

    def parse_template(self, local_results: list, top_n = 3, max_words=1000):
        '''
            Remove useless fields in local_results. Keep only the fields that are useful for GPT to summarize:

                {
                    "Name" (Required): name of the place,
                    "Rating": rating,
                    "Subtitle": subtitle in the template,
                    "BodyText": some detailed information.
                    "Additional": additional information (e.g. which website to book)
                }

            Inputs:
                top_n: number of places to summarize,
                max_words: maximum number of words (an estimate of # tokens) in the json list.

            The returned list will not exceed top_n results or max_words length, whichever is shorter.
        '''
        # preserve the order returned by Yelp
        summarized_results = []
        total_words_approx = 0

        for place in local_results:
            place = place["TemplateData"]
            try:
                curr_result = {
                    "Name": place["Title"],
                    "Rating": place["Rating"],
                    "Subtitle": place["Subtitle"],
                    "BodyText": place["BodyText"],
                    "Additional": place["Footer"],
                }
            except:
                continue
            summarized_results.append(curr_result)
            
            total_words_approx += len(json.dumps(curr_result).split())

            if len(summarized_results) >= top_n or total_words_approx >= max_words:
                break
        return summarized_results
    
    def google_map_local_search(self, query, radius=10000, lat=DEFAULT_REQ_INFO["Latitude"], lon=DEFAULT_REQ_INFO["Longitude"], rank_by_distance=False, top_n = 3, max_words=1000):
        # api doc: https://developers.google.com/maps/documentation/places/web-service/search-text
        '''
            Inputs:
                query: the text query to search for
                radius: Defines the distance (in meters) within which to return place results. maximum: 50000 (meters)
                lat: latitude of the user
                lon: longitude of the user
                rank_by_distance: whether we want to rank the output by distance, if True, radius will not be used. 

            Remove useless fields in local_results. Keep only the fields that are useful for GPT to summarize:
                {
                    "Name" (Required): name of the place,
                    "Location" (Required): only keep the address,
                    "Rating" (Optional): 0-5 rating score, sort from high to low in the list,
                    "Attributes" (Option): Other attributes for consideration.
                }
        '''
        GOOGLE_MAP_KEY = os.environ["GOOGLE_MAP_KEY"]
        endpoint = "https://maps.googleapis.com/maps/api/place/textsearch/json?"
        q = query.strip(string.punctuation).replace(" ", "%20")
        # for ranking:
        # prominence (default). This option sorts results based on their importance. Ranking will favor prominent places within the set radius over nearby places that match but that are less prominent. Prominence can be affected by a place's ranking in Google's index, global popularity, and other factors. When prominence is specified, the radius parameter is required. 
        # distance. This option biases search results in ascending order by their distance from the specified location. When distance is specified, one or more of keyword, name, or type is required and radius is disallowed.
        if rank_by_distance:
            link = endpoint + f"query={q}&rankby=distance&location={lat},{lon}&key={GOOGLE_MAP_KEY}"
        else:
            link = endpoint + f"query={q}&radius={radius}&location={lat},{lon}&key={GOOGLE_MAP_KEY}"
        local_results = requests.get(link).json()
        summarized_results = []
        total_words_approx = 0

        if local_results["status"] != "OK":
            return ["No result"]

        for place in local_results["results"]:
            try:
                if place["business_status"] != "OPERATIONAL":
                    continue
                curr_result = {
                    "Name": place["name"],
                    "Location": place["formatted_address"],
                    "Rating": place["rating"],
                }
            except:
                continue
            summarized_results.append(curr_result)
            
            total_words_approx += len(json.dumps(curr_result).split())

            if len(summarized_results) >= top_n or total_words_approx >= max_words:
                break
        return summarized_results

    def retrieve_useful_data(self, houndify_json, query):
        result = houndify_json["AllResults"][0]
        # base_msg = """Use the following information to answer the question: {query} in your style. You need to include as many choices and details as possible for the user. When you chat with me, pretend you are speaking like human. For all numerical values, convert them to text. For example "0.67" or "0. 67" should be "zero point six seven", "4.5" or "4. 5" should be "four and a half". When you see units like miles, convert them to metrics, like kilometers. \n {result}" + "\nAppend '\nSource: {src}' at the end of your response."""
        # kwargs = {"query": query,  "result": json.dumps(summarized_result), "src": "Houndify"}
        
        if "NativeData" in result and "LocalResults" in result["NativeData"]:
            local_results = result["NativeData"]["LocalResults"]
            summarized_result = self.parse_native(local_results, self.top_n)
            return json.dumps(summarized_result)
        elif "TemplateData" in result and "Items" in result["TemplateData"]:
            local_results = result["TemplateData"]["Items"]
            summarized_result = self.parse_template(local_results, self.top_n)
            return json.dumps(summarized_result)
        elif result["SpokenResponseLong"] != "Didn't get that!":
            return result["SpokenResponseLong"]
        else:
            # Houndify failed
            lat = self.client.HoundRequestInfo["Latitude"]
            lon = self.client.HoundRequestInfo["Longitude"]
            summarized_result = self.google_map_local_search(query, radius=10000, lat=lat, lon=lon, top_n=self.top_n)
            if len(summarized_result) == 1 and summarized_result[0] == "No results":
                return "No results"
            elif len(summarized_result) > 0:
                return json.dumps(summarized_result)
            else:
                return "No results"

class LocalSearchTool(BaseTool):
    name: str = "local_search"
    # description: str = "Search local restaurants, places of interests, etc. Also helpful for local weathers. This is not a tool for searching the location of a place. If the user is asking about history or general common knowledge (e.g. general introduction or history or location of a place), even if could be updated after the year 2021, you should not use it."
    description: str = "Search for local restaurants, places of interest and weather information. Not intended for searching the location of a specific place or for common knowledge questions."
    user_description: str = "You can enable this to search for local restaurants, places of interest etc. GPS must be enabled to use this tool"
    def __init__(self, func: Callable=None, **kwargs) -> None:
        top_n = kwargs.get(f"{self.name}_top_n", 3)
        self.ls = LocalSearch(user_id="test_user", req_info=DEFAULT_REQ_INFO, top_n=top_n)
        super().__init__(None)
        self.args["properties"]["req_info"]["description"] = "User's current location. If you don't know user's location, you should still include empty dict {} as req_info in the arguments"
        self.args["properties"]["query"]["description"] = "The string used to search. Make it as concise as possible"
        self.args["required"] = ["query", "req_info"]
    
    def on_enable(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def on_disable(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def _run(self, query: str, req_info: dict=None):
        return self.ls.search(query, req_info)
