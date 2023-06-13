# coding = utf-8
import requests
import json
from urllib3 import encode_multipart_formdata
from collections import OrderedDict

body = OrderedDict(
    [("mode",("get_gnss")),
     ("pointname",("2qxndb1300-2")),
     ("starttime",("2023-05-10 07:00:00")),
     ("endtime",("2023-05-11 12:00:00"))
     ]
)
boundary='_____________________'
m = encode_multipart_formdata(body, boundary=boundary)
url = 'https://cloud.yuejin360.com/index.php/home/webapi'   
appcode = "{\"appcode\":\"fbaf724a1a950591517151718d1c2f8f\"}"
headers = \
    {
    'appcode': appcode,
    'content-type': "multipart/form-data; boundary="+boundary
    }
response = requests.post(url, data=m[0],headers=headers)
resss = response.json()
print(resss)