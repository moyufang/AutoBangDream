from enum import Enum, auto
import requests as rq
import re
import os
import json

class Mode(Enum):
  FetchSheetsHeader = auto() # 爬取歌单头
  SpecialChar = auto()       # 预处理时，将原歌名中的特殊字符 special_char 都删掉
  FetchOne = auto()          # 爬取单首歌谱
  FetchLack = auto()         # 根据 sheets_header.json 和 sheets/*.bestdori 爬取缺少的歌谱

sheet_dir = './sheet/'
fetch_dir = './sheet/fetch/'
sheets_dir = './sheet/sheets/'
special_char = \
  r"[ /_\(\)\[\]\{\}!?:,.@#$%^&*=\"\'\-—（）【】！？：，。”“‘’「」　＆＊＝×·・★☆◎－∞♪\n\r\t↑]"

headers = {
  "user-agent": r"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
  "Requests":"",
  "Upgrade-Insecure-":"1",
  "Sec-Ch-Ua": r'"Google Chrome";v="125", "Chromium";v="125","Not.A/Brand";v="24"',
  "Sec-Ch-Ua-Mobile": "?0",
  "Sec-Ch-Ua-Platform": "Windows",
  "Sec-Fetch-User":"?1",
  "Sec-Fetch-Site":"same-origin",
  "Sec-Fetch-Mode":"navigate",
  "Sec-Fetch-Dest":"document",
  "Priority":"u=0,i",
  "Cookie":"token=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTA1NTY0LCJpYXQiOjE3MTY1OTU1NzIsImV4cCI6MTcxOTE4NzU3MiwiaXNzIjoiQmVzdGRvcmkvQXV0aCJ9.awpCcovCxaxIPEqxWV-iWQ5Kqp_KQpLkhI01iS9M5lPAqmj6kC7F9FopLZZqBHWiichGZOX7HHJ-0Wa-5caOXw; _gid=GA1.2.612823731.1717638853; _ga_W15VJ513VC=GS1.1.1717638852.85.1.1717638970.0.0.0; _ga=GA1.1.1286663543.1714844449",
  "Cache-Control":"max-age=0",
  "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
  "Accept-Encoding":"utf-8, gzip, deflate",
  "Accept-Language":"zh-CN,zh;q=0.9",
}

def get_sheet_url(sheet_id:int, diff:int):
  assert(0 <= int(diff) and int(diff) <= 4)
  diff = int(diff)
  map = ['easy', 'normal', 'hard', 'expert', 'special']
  return f"https://bestdori.com/api/charts/{sheet_id}/{map[diff]}.json"

def get_all_header_url():
  return r"https://bestdori.com/api/songs/all.5.json"

def get_rp(mode:Mode, url:str, file_path:str, json_indent:int=2, is_breif:bool=True):
  rp = rq.get(url=url, headers=headers)
  print(f"Mode: {mode} Fetching \"{url}\" ...")
  if not is_breif:
    print("response:")
    print("\tstatus_code", rp.status_code)
    print("\tencoding", rp.encoding)
    print("\tcookies", rp.cookies)
    print("\tencoding", rp.headers.get('Content-Encoding'))
  if rp.status_code != 200:
    print("Fetching failed with: ", file_path)
    exit(0)
  data = json.loads(rp.content.decode('utf-8'))
  with open(file_path, "w") as file: json.dump(data, file, indent=json_indent)
  if not is_breif:
    print(f"Saved as \"{file_path}\"")
    print()
  return rp, data

def shave_str(title:str):
  return re.sub(special_char, '', title).lower()

def fetch_sheets_header():
  url = get_all_header_url()
  file_path = fetch_dir+"raw_sheets_header.json"
  rp, data = get_rp(Mode.FetchSheetsHeader, url, file_path)
  
  sheets_header = {}
  for sheet_id in data:
    item = data[sheet_id]
    diff = item['difficulty']
    titles = item['musicTitle']
    title = None
    sorted_titles = [titles[3], titles[0], titles[1]]
    for t in sorted_titles:
      if t != None: title = t; break
    if title == None:
      print(sheet_id, title, titles)
    title = shave_str(title)
    
    sheets_header[sheet_id] = [
      int(sheet_id),
      title,
      int(diff['0']['playLevel']),
      int(diff['1']['playLevel']),
      int(diff['2']['playLevel']),
      int(diff['3']['playLevel']),
      int(diff['4']['playLevel'] if '4' in diff else -1)
    ]
  file_path = sheet_dir + 'sheets_header.json'
  with open(file_path, "w", encoding='utf-8') as file:
    file.write('{\n')
    for k in sheets_header:
      file.write(f"\"{k}\":"+json.dumps(sheets_header[k])+',\n')
    file.write('"-1":[-1, "", -1, -1, -1, -1, -1]\n}')
  print(f"Saved as \"{file_path}\"")
  print()
  
def fetch_one(song_id:int, level:int):
  url = get_sheet_url(song_id, level)
  file_path = fetch_dir+"fetch_one_sheet_%d_%d.bestdori"%(song_id, level)
  rp, data = get_rp(Mode.FetchOne, url, file_path)
  
def fetch_lack():
  with open(sheet_dir+"sheets_header.json", "r", encoding="utf-8") as file:
    sheets_header = json.load(file)
  for k in sheets_header:
    item = sheets_header[k]
    if item[0] == -1: continue
    for i in range(0, 5):
      if item[i+2] == -1: continue
      sheet_bestdori = sheets_dir+'%d_%d.bestdori'%(item[0], i)
      if os.path.isfile(sheet_bestdori): continue
      
      url = get_sheet_url(item[0], i)
      file_path = sheets_dir+"%s_%s.bestdori"%(item[0], i)
      rp, data = get_rp(Mode.FetchLack, url, file_path, is_breif=True)


if __name__ == '__main__':
  mode = Mode.FetchSheetsHeader
  if mode == Mode.FetchSheetsHeader:
    fetch_sheets_header()
  elif mode == Mode.SpecialChar:
    title = "[期間限定 SPECIAL]ピコっと！パピっと！！ガルパ☆ピコ！！！"
    print(title, "-->")
    print(shave_str(title))
  elif mode == Mode.FetchOne:
    fetch_one(306, 4)
  elif mode == Mode.FetchLack:
    fetch_lack()
        