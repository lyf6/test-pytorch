# coding = utf-8
import requests
import json



url = 'https://cloud.yuejin360.com/index.php/home/webapi'

body  = "-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\nget_gnss\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"pointname\"\r\n\r\n2qxndb1300-2\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"starttime\"\r\n\r\n2023-05-10 07:00:00\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"endtime\"\r\n\r\n2023-05-11 12:00:00\r\n-----011000010111000001101001--\r\n\r\n"
        # '-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\nget_gnss\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"pointname\"\r\n\r\n2qxndb1300-2\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"starttime\"\r\n\r\n2023-05-10 07:00:00\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"endtime\"\r\n\r\n2023-05-11 12:00:00\r\n-----011000010111000001101001--\r\n\r\n'
        # '-----011000010111000001101001\r\nContent-Disposition: form-data; name="mode"\r\n\r\nget_gnss\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name="pointname"\r\n\r\n2QXDB1270-1\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name="starttime"\r\n\r\n2023-05-1 07:00:00\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name="endtime"\r\n\r\n2023-05-11 07:00:00\r\n-----011000010111000001101001--\r\n'
# headers = dict()
appcode = "{\"appcode\":\"fbaf724a1a950591517151718d1c2f8f\"}"
# headers.setdefault("Content-Type","application/json;charset=UTF-8")
# headers.setdefault("sign",sign)
# headers.setdefault("appcode",appcode)
headers = \
    {
    'appcode': appcode,
    'content-type': "multipart/form-data; boundary=---011000010111000001101001"
    }


response = requests.post(url,data = body,headers=headers)
# headers={'appcode': {}.format(appcode)}
# cookies = {'PHPSESSID': 'w7n9kf9s5gdgm2jrllpOhin02'}
# res = requests.post(url,headers=headers,data=body)
# res = requests.post(url,,json=body)
# print(response.request.body.decode('utf-8'))
resss = response.json()
#r = requests.post(url,headers=headers,json=body)
print('ggg')
