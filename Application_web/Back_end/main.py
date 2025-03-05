from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deal_input import get_embeddings, use_model, load_model


# Modelo para validar la entrada del usuario
class InputModel(BaseModel):
    text: str

# Crear una instancia de FastAPI
app = FastAPI()

# Permitir solicitudes CORS desde orígenes específicos
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5050",
    'https://aihumantext.netlify.app',
    'https://ai-vs-human-6qv7.onrender.com'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/request_model")
async def request_model(input_data: InputModel):
    try:
        user_input = input_data.text.strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="Empty text")

        model = load_model()
        embeddings_text = get_embeddings(user_input)
        soft_max_output = use_model(embeddings_text, model)

        return {"human": float(round(soft_max_output[0], 2)), "machine": float(round(soft_max_output[1], 2))}
    
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Validaton error: {str(ve)}")
    
    except TypeError as te:
        raise HTTPException(status_code=500, detail=f"Data type error: {str(te)}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error : {str(e)}")




