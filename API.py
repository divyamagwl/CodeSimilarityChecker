import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse
from typing import Optional
from io import BytesIO

app = FastAPI()

@app.post("/upload-file/")
async def upload_file( max_level: int,
                       loop_construct: int=Query(0,enum=[0,1]),
                       if_construct: int=Query(0,enum=[0,1]),
                       control_construct: int=Query(0,enum=[0,1]),
                       function_construct: int=Query(0,enum=[0,1]),
                       arithOp_construct: int=Query(0,enum=[0,1]),
                       excep_construct: int=Query(0,enum=[0,1]),
                      file1: UploadFile = File(...),
                      file2: UploadFile = File(...)):
    
    # save both the input files locally
    file_location = f"./files/{file1.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file1.file.read())
        
    file_location = f"./files/{file2.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file2.file.read())
        
    construct_flag = f"{loop_construct}" + f"{if_construct}" + \
                     f"{control_construct}" + f"{function_construct}" + \
                     f"{function_construct}" + f"{arithOp_construct}" + \
                     f"{excep_construct}"
    
    # run the code similarity checker for the saved input files
    ast_filePath = 'similarity.py'
    os.system(f'python3 {ast_filePath} ./files/{file1.filename} ./files/{file2.filename} {construct_flag} {max_level} > SimilarityScore.txt')
    
    # return the txt file as response
    return FileResponse("SimilarityScore.txt")

if __name__ == "__main__":
    uvicorn.run(app, port=5500)