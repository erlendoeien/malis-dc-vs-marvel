import aiohttp
import asyncio
import copy
from io import BytesIO
from PIL import Image


def initialize():
    dc, marvel = [{"name": "batman-2016", "chapter": 1, "imgs": {}},{"name": "the-flash-2016", "chapter":1, "imgs":{}}], [{"name": "immortal-hulk-2018", "chapter": 1, "imgs": {}}] 
    return dc, marvel


def get_url(comic_name: str, chapter: int, page: int):
    str_page = "0" + str(page) if page < 10 else page
    return f"https://readcomicsonline.ru/uploads/manga/{comic_name}/chapters/{chapter}/{str_page}.jpg"


async def get_comic_page(
    session: aiohttp.ClientSession, name: str, chapter: int, page: int, **kwargs
) -> bytes:
    url = get_url(name, chapter, page)
    print(f"Requesting {url}")
    resp = await session.request("GET", url=url, **kwargs)
    if resp.status == 200:
        print(f"Received response for {url}")
        data: bytes = await resp.read()
        return data
    return None

def stream_to_img(stream: bytes):
    return Image.open(BytesIO(stream)) if stream else None


async def get_comic_book(comic: dict, session: aiohttp.ClientSession, **kwargs):
    # TODO: Mutating the comic-dict -> Refactor all code to be non-mutuable
    page_tasks = []
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
    for task in asyncio.as_completed(page_tasks):
        try:
            response = await task
            if not response:
                continue
            comic["imgs"][response] = response
        except Exception as e:
            print(f"Last page before exception {page-1}")
            print(e)


async def get_comics(dc_comics: list, marvel_comics: list, **kwargs):
    dc, marvel = copy.deepcopy(dc_comics), copy.deepcopy(marvel_comics)
    async with aiohttp.ClientSession() as session:
        for comic in dc:
            await get_comic_book(comic=comic, session=session, **kwargs)
        for comic in marvel:
            await get_comic_book(comic=comic, session=session, **kwargs)
    return dc, marvel


async def fetch_data(**kwargs):
    dc, marvel = initialize()
    dc, marvel = await get_comics(dc_comics=dc, marvel_comics=marvel, **kwargs)
    return dc, marvel


def save_images(comics: list, dir_name: str):
    imgs = [stream_to_img(stream) for stream in comics[0]["imgs"]]
    for idx, img in enumerate(imgs, start=1):
        if img:
            img.save(f"./project/data/{dir_name}/page_async_{idx}.jpg")


async def main():
    dc_comics, marvel_comics = await fetch_data(ssl=False)
    save_images(dc_comics, "DC")
    save_images(marvel_comics, "Marvel")



if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
