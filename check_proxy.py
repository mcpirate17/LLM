import requests
resp = requests.get("http://127.0.0.1:8091/api/v1/components", params={'status':'approved'})
print(resp.status_code)
print(len(resp.json()))
