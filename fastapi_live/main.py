from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel
from sqlmodel import Field, SQLModel, create_engine, Session, select

app = FastAPI()


names = {
    "Waldemar": 178,
    "Elias": 200,
    "Johannes": 150
}

persons = []

class Person(str, Enum):
    elias, waldemar = "Elias", "Waldemar"
    # waldemar = 

class PersonSchema(SQLModel, table=True):
    id: int = Field(primary_key=True, default=None, exclude=True)
    first_name: str
    last_name: str
    age: int
    height: float | None = None


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=True)

SQLModel.metadata.create_all(engine)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/greet/name/{name}")
async def greet(name: str):
    return {"message": f"Hello {name}"}

@app.get("/greet/age/{age}")
async def get_age(age: int):
    return {"message": f"Your age is {age}"}

@app.get("/greet/person/{person}")
def greet_person(person: Person):
    return {"message": f"Hello {person.value}"}

@app.get("/person")
def get_person(name: str):
    return {f"{name}'s height is {names.get(name, 'Unknown')}"}


@app.get("/person/list_all")
def list_persons(max_height: float | None = None):
    with Session(engine) as session:
        statement = select(PersonSchema)
        if max_height:
            statement = statement.where(PersonSchema.height <= max_height)
        persons = session.exec(statement).all()
    return persons

@app.post("/person")
def add_person(person: PersonSchema):
    with Session(engine) as session:
        session.add(person)
        session.commit()
    return {"message": f"successfully added person"}



# if __name__ == "__main__":
#     app.run()