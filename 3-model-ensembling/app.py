from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
from driver_pkg.driver import driver_main
import pandas as pd
from fastapi.concurrency import run_in_threadpool

app = FastAPI()

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    temp_input_path = Path("driver_pkg/tmp_files/input.csv")
    with temp_input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    driver_main([
        "--input-csv", str(temp_input_path),
        "--model-pkl",  "./2-ensembleModel/logesticRegression/model.pkl",
        "--label-map",  "./2-ensembleModel/logesticRegression/model_label_encoder.pkl",
    ])
    output_path = Path("./driver_pkg/tmp_files/output.csv")
    result_df = pd.read_csv(output_path)
    result_df.replace([float('inf'), float('-inf')], 0, inplace=True)
    result_df.fillna(0, inplace=True)
    result_json = result_df.to_dict(orient="records")
    return {"prediction": result_json}
