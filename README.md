# Marvel vs DC

This is a course project for to learn

1. How to define a problem suitable for machine learning
2. How to utilise different models to create solutions for the problem
3. Try to solve the problem

We will attempt to create a model which can distinguish the art styles of Marvel comics from DC comics.

# How to use `get_data.py`
1. Make sure you have python installed (tested with v3.8.3), and preferably a virtual environemnt
2. Install dependencies, e.g. with pip: `pip install -r requirements.txt`
3. Find the comic book you would like to fetch, from https://readcomicsonline.ru/
4. Get the name and the chapter (edition) of the comic from the URL, e.g. `immortal-hulk-2018` and chapter `1` in the url
https://readcomicsonline.ru/comic/immortal-hulk-2018/1/1.
The URLs are always on the format `https://readcomicsonline.ru/comic/<comic-name>/<chapterNr>/<pageNr>`.
There are some examples in the script on the format the comic book name and chapter should be added to the script
5. Run the script and await for the images to be downloaded
