import requests
res_reg = requests.post('http://localhost:8000/api/register', json={'username': 'test2', 'email': 'test2@test.com', 'password': 'pw'})
res = requests.post('http://localhost:8000/api/login', json={'email': 'test2@test.com', 'password': 'pw'})
token = res.json().get('access_token')

text = '''Online disinformation and threats made campaigning for Jersey's election challenging this year.

Re-elected Constable of St Peter, Richard Vibert said he thought "there was more negativity" on social media this time than during the last election in 2022.

Deputy of St Brelade, Gabriel Raimondo, said the online abuse he received "took its toll".'''

res2 = requests.post('http://localhost:8000/api/predict', json={'text': text}, headers={'Authorization': 'Bearer ' + token})
print(res2.status_code, res2.text)
