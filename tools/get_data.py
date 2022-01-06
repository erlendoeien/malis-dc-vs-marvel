import aiohttp
import asyncio
import copy
import os
from io import BytesIO
from PIL import Image

DC_COMICS_NAMES = [
    # "nightwing-2016",
    # "the-hellblazer-2016",
    # "batgirl-2016",
    # "superman-2016",
    # "red-hood-and-the-outlaws-2016",
    # "wonder-woman-2016",
    # "aquaman-2016-rebirth",
    # "supergirl-2016",
    # "green-lanterns-2016",
    # "blue-beetle-2016",
    # "harley-quinn-2016",
    # "green-arrow-2016",
    # "teen-titans-2016",
    # "cyborg-2016",
    # "deathstroke-2016",
    # "heman-thundercats-2016",
    # "suicide-squad-2016",
    # "justice-league-2016",
]
MARVEL_COMICS_NAMES = [
    # "the-thing-and-the-human-torch-by-dan-slott-2018",
    # "absolute-carnage-2019/",
    # "allnew-wolverine-2015",
    # "amazing-spiderman-2018",
    # "black-widow-widows-sting-2020",
    # "hawkeye-kate-bishop-2021",
    # "death-of-doctor-strange-avengers-2021",
    # "ms-marvel-infinity-comic-primer-2021",
    # "captain-america-heroes-return-the-complete-collection-2021",
    # "hulk-2021",
    # "thanos-2016",
    # "new-mutants-2019",
    # "the-mighty-thor-2015",
    # "deadpool-2015",
    # "daredevil-2019",
    # "spidergwen-vol-2-2015",
    # "xforce-2019",
    # "the-darkhold-black-bolt-2021",
    # "shangchi-brothers-sisters-infinity-comic-2021",
]


def initialize():
    """Hardcoded initialize function where one can supply
    the wanted comic books by name and their 'chapter' (edition)"""
    dc = [{"name": name, "chapter": 1, "imgs": []} for name in DC_COMICS_NAMES]
    marvel = [
        {"name": name, "chapter": 1 if "TBP" not in name else "TBP", "imgs": []}
        for name in MARVEL_COMICS_NAMES
    ]
    return dc, marvel


def get_url(comic_name: str, chapter: int, page: int):
    """Maps the comic book page to the correct image url. The comic book name is retrieved from the url
    of the targeted comic book."""
    str_page = "0" + str(page) if page < 10 else page
    return f"https://readcomicsonline.ru/uploads/manga/{comic_name}/chapters/{chapter}/{str_page}.jpg"


async def get_comic_page(
    session: aiohttp.ClientSession, name: str, chapter: int, page: int, **kwargs
) -> bytes:
    """Fetches a single comic book page, asynchronous"""
    url = get_url(name, chapter, page)
    print(f"Requesting {url}")
    resp = await session.request("GET", url=url, **kwargs)
    if resp.status == 200:
        print(f"Received response for {url}")
        data: bytes = await resp.read()
        return data
    return None


def stream_to_img(stream: bytes):
    """Converts a image byte stream to a PIL Image-object"""
    return Image.open(BytesIO(stream))


async def get_comic_book(comic: dict, session: aiohttp.ClientSession, **kwargs):
    # TODO: Mutating the comic-dict -> Refactor all code to be non-mutuable
    copy_comic = copy.deepcopy(comic)
    page_tasks = []
    # Creates the async takes to be executed
    for page in range(1, 50):  # assuming maximum comic book length
        page_tasks.append(
            get_comic_page(
                session=session,
                name=comic["name"],
                chapter=comic["chapter"],
                page=page,
                **kwargs,
            )
        )
    # Iterates over each async task to store only valid pages
    for task in asyncio.as_completed(page_tasks):
        try:
            response = await task
            if response:
                copy_comic["imgs"].append(response)
        except Exception as e:
            print(f"Last page before exception {page-1}")
            print(e)
    return copy_comic


async def get_comics(comics: list, session: aiohttp.ClientSession, **kwargs):
    """Fetches each page for each comic book dictionary supplied in the list"""
    comics_copy = []
    for comic in comics:
        comics_copy.append(await get_comic_book(comic=comic, session=session, **kwargs))
    return comics_copy


async def get_all_comics(dc_comics: list, marvel_comics: list, **kwargs):
    """Fetches each page for each comic book dictionary supplied
    As it is suggested to use the same aiohttp-session, there is some red"""
    dc, marvel = [], []
    async with aiohttp.ClientSession() as session:
        dc = await get_comics(dc_comics, session, **kwargs)
        marvel = await get_comics(marvel_comics, session, **kwargs)
    return dc, marvel


async def fetch_data(**kwargs):
    """Initializes the targeted comic books and fetches them"""
    dc, marvel = initialize()
    async with aiohttp.ClientSession() as session:
        dc = await get_comics(dc, session, **kwargs)
        marvel = await get_comics(marvel, session, **kwargs)
    return dc, marvel


def save_images(comics: list, dir_name: str):
    """Stores the comic books in directories by their comic book name,
    one JPG-file for each comic book page.
    NB: Often the 1-2 first and last pages are not actual 'comics'
    and might need to be excluded"""
    for comic in comics:
        imgs = [stream_to_img(stream) for stream in comic["imgs"] if not None]
        dir_path = os.path.join(
            os.path.realpath("data"), dir_name, f"{comic['name']}-{comic['chapter']}"
        )
        os.makedirs(dir_path, exist_ok=True)
        for idx, img in enumerate(imgs, start=1):
            if img:
                img.save(os.path.join(dir_path, f"page_async_get_{idx}.jpg"))


def main():
    dc_comics, marvel_comics = asyncio.get_event_loop().run_until_complete(
        fetch_data(ssl=False)
    )
    save_images(dc_comics, "DC")
    save_images(marvel_comics, "Marvel")


if __name__ == "__main__":
    main()
