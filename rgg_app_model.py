from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import pickle
import pandas as pd
import sqlite3

model = pickle.load(open('model/advertising_model.pkl','rb'))
class RawData(BaseModel):
    data: list[list[int]]
class Prediction(BaseModel):
    data: list[list[float,float,float]]
class NewData(BaseModel):
    tv: int
    radio: int
    newspaper: int
    sales: int
class NewDataList(BaseModel):
    data: list[NewData]

conn = sqlite3.connect('data/database.db')
cursor = conn.cursor()

app = FastAPI()
@app.get('/')
async def home():
    return "API for ML model prediction usage by Raúl García"


# 1. Endpoint de predicción
@app.post('/predict')
async def prediction(pred:Prediction):
    input_data = pred.data[0]
    result = model.predict([[input_data[0], input_data[1], input_data[2]]])
    return {"prediction": result[0]}

# 2. Endpoint de ingesta de datos
@app.post('/ingest')
async def new_data(data: RawData):
    for record in data.data:  # Itera sobre las listas internas
        if len(record) != 4:
            raise HTTPException(status_code=400, detail="Each record must have exactly 4 elements")
        cursor.execute(
            '''
            INSERT INTO advertising (TV, radio, newspaper, sales)
            VALUES (?, ?, ?, ?)
            ''',
            (record[0], record[1], record[2], record[3])
        )
    conn.commit()
    return {'message': 'Datos ingresados correctamente'}

# 2. Endpoint de reentramiento del modelo
@app.post('/retrain')
async def retrain():
    cursor.execute(
        '''
        SELECT * FROM advertising;
        '''
    )
    results = cursor.fetchall()
    res_dict={'TV':[],'radio':[],'newspaper':[],'sales':[]}
    for result in results:
        res_dict['TV'].append(result[0])
        res_dict['radio'].append(result[1])
        res_dict['newspaper'].append(result[2])
        res_dict['sales'].append(float(result[3]))
    df_res = pd.DataFrame(res_dict)
    x1_train = df_res.drop(columns='sales')
    y1_train = df_res['sales']
    model.fit(x1_train,y1_train)
    pickle.dump(model,open('./model/advertising_model.pkl','wb'))
    return {'message': 'Modelo reentrenado correctamente.'} 

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)