import requests
request=requests.get('http://api.open-notify.org/iss-now.json')
print(request.status_code)
print(request.text)
request=requests.get('http://api.open-notify.org/astros.json')
print(request.status_code)
print(request.text)