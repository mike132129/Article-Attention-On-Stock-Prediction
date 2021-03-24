import pymongo
client = pymongo.MongoClient("mongodb://localhost:27017/")
DB = client["newsdb"]

# Collections
NEWS = DB["tsmc"]


