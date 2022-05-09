import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Optional
from io import BytesIO

app = FastAPI()

@app.post("/upload-file/")
async def create_upload_file(file1: UploadFile = File(...),
                             file2: UploadFile = File(...)):
    file_location = f"./files/{file1.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file1.file.read())
        
    file_location = f"./files/{file2.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file2.file.read())
    
    ast_filePath = '/home/aditya/Desktop/college_sems/6/PL/project/CodeSimilarityChecker/generateAST.py'
    os.system(f'python3 {ast_filePath} ./files/{file1.filename} ./files/{file2.filename} > temp.txt')
    
    # return {"info": f"file '{uploaded_file.filename}' saved at '{file_location}'"}


if __name__ == "__main__":
    uvicorn.run(app, port=5500)

# os.system('ls -l > temp.txt')
