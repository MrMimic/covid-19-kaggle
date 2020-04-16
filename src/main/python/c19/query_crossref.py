import requests
import urllib

headers = {'User-Agent': 'covid_data (mailto:baptistemetge@gmail.com)'}


def get_data_from_title(title):
    try:
        q = urllib.parse.quote(title)
        url = "https://api.crossref.org/works?query.bibliographic=" + q
        print(url)
        response = requests.get(url, headers=headers)
        data = response.json()
        return data
    except Exception as e:
        print(e)


# EXEMPLE
# paper  =get_data_from_title('Early Transmission Dynamics in Wuhan, China, of Novel Coronavirus-Infected Pneumonia')
# print(paper['message']['items'][0]['DOI'])
# print(paper['message']['items'][0]['title'])
