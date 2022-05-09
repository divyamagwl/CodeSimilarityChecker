import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from typing import Optional
from io import BytesIO

app = FastAPI()

@app.post("/upload-file/")
async def upload_file(file1: UploadFile = File(...),
                             file2: UploadFile = File(...)):
    
    # save both the input files locally
    file_location = f"./files/{file1.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file1.file.read())
        
    file_location = f"./files/{file2.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file2.file.read())
    
    # run the code similarity checker for the saved input files
    ast_filePath = 'generateAST.py'
    os.system(f'python3 {ast_filePath} ./files/{file1.filename} ./files/{file2.filename} > temp.txt')
    
    # return the txt file as response
    return FileResponse("SimilarityScore.txt")

if __name__ == "__main__":
    uvicorn.run(app, port=5500)