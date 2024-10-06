from fastapi import FastAPI
from xgboost import Booster
from fastapi.responses import ORJSONResponse


from helpers import lifespan
from helpers.inputs import Input

app = FastAPI(
    lifespan=lifespan.lifespan,
    title="Problematic Internet Use API",
)


@app.post("/predict/")
async def predict(input_: Input) -> ORJSONResponse:
    if isinstance(lifespan.model, Booster):
        inputs = input_.to_dmatrix()
    else:
        inputs = input_.to_pandas()
    prediction_probs = lifespan.model.predict(inputs)
    prediction = int(prediction_probs[0] > 0.5)
    return ORJSONResponse(
        content={"prediction": prediction, "probability": float(prediction_probs[0])}
    )
