import datetime
import pymongo


# Get Mongo Database URL Connection String
file = open("srv.txt")
srv = file.read()
file.close()

client = pymongo.MongoClient("{}".format(srv))

food = client.food
records = food.records

print("Food input")
food = str(input())
timestamp = datetime.datetime.now()

data = {
    'food' : food, 
    'time' : timestamp
}

try: 
    records.insert_one(data)
except Exception as e: 
    print(e)
    pass