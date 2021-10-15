import aiohttp
import asyncio
from io import BytesIO
from PIL import Image
import ssl

# ssl_ctx = ssl.create_default_context(cafile='/path_to_client_root_ca')
# ssl_ctx.load_cert_chain('/path_to_client_public_key.pem', '/path_to_client_private_key.pem')

# conn = aiohttp.TCPConnector(ssl_context=ssl_ctx)

def fetch_data():
    url = "https://readcomicsonline.ru/uploads/manga/2020-force-works-2020/chapters/1/04.jpg"
    response = requests.get(url)
    byte_stream = BytesIO(response.content)
    img = Image.open(byte_stream)
    img.save("test.jpg")

def initialize():
    dc, marvel = [{"name": "batman-2016", "chapter": 1, "imgs": {}}], []
    return dc, marvel

def get_url(comic_name:str, chapter: int, page: int):
    str_page ="0"+str(page) if page < 10 else page
    return f"https://readcomicsonline.ru/uploads/manga/{comic_name}/chapters/{chapter}/{str_page}.jpg"

async def get_comic_page(
    session: aiohttp.ClientSession,
    name: str,
    chapter: int,
    page: int,
    **kwargs
) -> bytes:
    url = get_url(name, chapter, page)
    print(f"Requesting {url}")
    resp = await session.request('GET', url=url, **kwargs)
    # Exceptions are passed through for non-2xx responses
    print(f"Received response for {url}")
    if resp.status > 200:
        return None
    data: bytes = await resp.read()
    return data

def stream_to_img(stream: bytes):
    return Image.open(BytesIO(stream)) if stream else None

async def main( **kwargs):
    # Asynchronous context manager.  Prefer this rather
    # than using a different session for each GET request
    dc, marvel = initialize()
    async with aiohttp.ClientSession() as session:
        for comic in dc:
            page_tasks = []
            for page in range(1,150): # assuming maximum comic book length
                page_tasks.append(get_comic_page(session=session, name=comic["name"], chapter=comic["chapter"], page=page, **kwargs))
            for task in asyncio.as_completed(page_tasks):
                try:
                    response = await task
                    if not response:
                        continue
                    comic["imgs"][response] = response
                except Exception as e:
                    # Expecting 404 error when indexing out of bounds for the pages
                    print(f"Last page before exception {page-1}")
                    print(e)
    return dc#, marvel
        # asyncio.gather() will wait on the entire task set to be
        # completed.  If you want to process results greedily as they come in,
        # loop over asyncio.as_completed()
        # htmls = await asyncio.gather(*tasks, return_exceptions=True)
        # return htmls



if __name__ == "__main__":
    dc_comics =  asyncio.get_event_loop().run_until_complete(main(ssl=False))#, marvel_stream = main()
    # print(dc_comics)
    dc_imgs = [stream_to_img(stream) for stream in dc_comics[0]["imgs"]]
    # dc_imgs = [stream_to_img(comic["imgs"][i]) for i, comic in enumerate(dc_comics, start=1) if not None]
    print(dc_imgs)
    for idx, img in enumerate(dc_imgs, start=1):
        if img:
            img.save(f"./test_imgs/test_{idx}.jpg")
    