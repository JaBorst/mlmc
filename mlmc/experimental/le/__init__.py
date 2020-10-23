from .LRDGraphEmbedding import LRDG

def get_model_from_omniboard(id, artifact, db):
    import requests
    url="https://aspra29.informatik.uni-leipzig.de:8000/file"
    payload = {'db_name': db, 'id': id, "artifact": artifact}
    with open("tmp.file", "wb") as file:
        r = requests.get(url,payload,verify=False)
        file.write(r.content)
    return "tmp.file"
