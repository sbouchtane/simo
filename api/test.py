import requests

url = "http://localhost:8000/api/vasicek/"

data = {
    "target_date": "01/06/2023",
    "NR": 100,
    "MAP": 9,
    "k": 10
}

response = requests.post(url, data=json.dumps(data), headers={
                         'Content-Type': 'application/json'})

print(response.json())
