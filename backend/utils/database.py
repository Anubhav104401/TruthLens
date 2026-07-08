import asyncio

class MockCollection:
    def __init__(self, name):
        self.name = name
        self.data = []
        
    async def find_one(self, query):
        for item in self.data:
            match = True
            for k, v in query.items():
                if item.get(k) != v:
                    match = False
                    break
            if match:
                return item
        return None
        
    async def insert_one(self, doc):
        from bson import ObjectId
        doc["_id"] = ObjectId()
        self.data.append(doc)
        return type('InsertOneResult', (), {'inserted_id': doc["_id"]})()

    def find(self, query):
        class Cursor:
            def __init__(self, data):
                self.data = data
            def sort(self, key, direction):
                return self
            def limit(self, num):
                return self
            async def to_list(self, length):
                return self.data[:length]
        
        # Filter by query
        filtered_data = []
        for item in self.data:
            match = True
            for k, v in query.items():
                if item.get(k) != v:
                    match = False
                    break
            if match:
                filtered_data.append(item)
        # Reverse to simulate sorting by timestamp descending roughly
        return Cursor(filtered_data[::-1])

# In-memory collections
users_collection = MockCollection("users")
predictions_collection = MockCollection("predictions")
feedback_collection = MockCollection("feedback")

