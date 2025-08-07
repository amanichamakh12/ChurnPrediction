from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pydantic import BaseModel

import pandas as pd
import numpy as np
import shutil
import subprocess
import traceback
import requests
import pickle
import os

app = FastAPI()


class ClientFeatures(BaseModel):
    CLI_id: int
    NBENF: int
    SEG: int
    nb_transactions: int
    montant_total: float
    montant_moyen: float
    montant_max: float
    montant_min: float
    dernier_montant: float
    nb_types_produits: int
    nb_libelles_produits: int
    age: int
    anciennete: int
    SEXT: str



with open("C:/Users/user/Downloads/ChurnPrediction/model_rf.pkl", "rb") as f:
    model = pickle.load(f)



def save_client(client: ClientFeatures):
    url = "http://localhost:8090/clients/save"
    client_json = jsonable_encoder(client)
    response = requests.post(url, json=client_json)
    return response.text

def send_backend(client: ClientFeatures, prediction: int, proba: float, reason: str,recommandation:str):

    save_client(client)
    url = "http://localhost:8090/predictions/predict"
    payload = {
        "client": jsonable_encoder(client),
        "predictionValue": "CHURN" if prediction == 1 else "NON-CHURN",
        "probability": round(proba, 2),

        "reason": reason,
        "recommandation":recommandation
    }
    response = requests.post(url, json=payload)
    return response.text
def generer_recommandation(causes):
    recommandations = {
        "nb_transactions": "Proposer une campagne de réengagement avec bonus",
        "montant_total": "Offrir une remise ou un avantage fidélité",
        "nb_types_produits": "Suggérer des produits personnalisés selon le profil",
        "montant_moyen": "Présenter des offres plus attractives ou adaptées",
        "anciennete": "Mettre en place une offre de fidélisation spéciale nouveaux clients",
        "nb_libelles_produits": "Envoyer un catalogue de produits diversifiés",
        "SEG": "Proposer un conseiller personnalisé pour ce segment",
        "SEXT": "Adapter l'offre marketing selon le profil du genre",
        "montant_max": "Faire découvrir des produits haut de gamme",
        "dernier_montant": "Inciter par une offre promotionnelle",
    }
    actions = [recommandations.get(c, "Revue manuelle recommandée") for c in causes]
    return "; ".join(actions)


def generer_recommandation(causes):
    recommandations = {
        "nb_transactions": "Proposer une campagne de réengagement avec bonus",
        "montant_total": "Offrir une remise ou un avantage fidélité",
        "nb_types_produits": "Suggérer des produits personnalisés selon le profil",
        "montant_moyen": "Présenter des offres plus attractives ou adaptées",
        "anciennete": "Mettre en place une offre de fidélisation spéciale nouveaux clients",
        "nb_libelles_produits": "Envoyer un catalogue de produits diversifiés",
        "SEG": "Proposer un conseiller personnalisé pour ce segment",
        "SEXT": "Adapter l'offre marketing selon le profil du genre",
        "montant_max": "Faire découvrir des produits haut de gamme",
        "dernier_montant": "Inciter par une offre promotionnelle",
    }

    actions = [recommandations.get(c, "Revue manuelle recommandée") for c in causes]
    return "; ".join(actions)



def predict_client(client: ClientFeatures):
    input_data = np.array([[  
        client.NBENF, client.SEG, client.nb_transactions, client.montant_total,
        client.montant_moyen, client.montant_max, client.montant_min,
        client.dernier_montant, client.nb_types_produits, client.nb_libelles_produits,
        client.age, client.anciennete,
        1 if client.SEXT.upper() == "M" else 0
    ]])

    prediction = int(model.predict(input_data)[0])
    proba = model.predict_proba(input_data)[0][1]

    features = [
        "NBENF", "SEG", "nb_transactions", "montant_total", "montant_moyen",
        "montant_max", "montant_min", "dernier_montant", "nb_types_produits",
        "nb_libelles_produits", "age", "anciennete", "SEXT"
    ]

    importances = model.feature_importances_
    top_indices = np.argsort(importances * input_data[0])[::-1][:4]
    causes = [features[i] for i in top_indices]

    explications = {
        "NBENF": "Le client pourrait bénéficier de produits adaptés aux familles.",
        "SEG": "Le client appartient à un segment considéré à risque, nécessitant une attention particulière.",
        "nb_transactions": "Le client effectue peu de transactions, ce qui peut refléter un faible engagement.",
        "montant_total": "Le volume global de dépenses du client est faible, suggérant un manque d’utilisation des services bancaires.",
        "montant_moyen": "Les montants dépensés sont faibles, ce qui peut indiquer un manque de services attractifs ou adaptés.",
        "montant_max": "Le client n’a jamais réalisé de transaction significative, ce qui peut indiquer un désintérêt pour les produits premium.",
        "montant_min": "Les opérations du client sont très faibles, traduisant une activité bancaire minimale.",
        "dernier_montant": "Le dernier montant payé était peu élevé, pouvant révéler une réduction de l’activité récente.",
        "nb_types_produits": "Le client utilise peu de types de produits, ce qui peut indiquer un manque de diversification dans l’offre.",
        "nb_libelles_produits": "La variété des produits utilisés est limitée, ce qui peut refléter un manque d’incitation à explorer d’autres services.",
        "age": "La tranche d’âge du client pourrait nécessiter une approche personnalisée.",
        "anciennete": "Le client est relativement nouveau, ce qui peut nécessiter des actions de fidélisation renforcées.",
        "SEXT": "Certaines préférences liées au genre peuvent nécessiter une adaptation de l’offre marketing."
    }

    reason = ""

    recommandation = ""

    if prediction == 1:
        reason = ", ".join([explications[cause] for cause in causes])
        recommandation = generer_recommandation(causes)

    send_backend(client, prediction, proba, reason,recommandation)

    return {
        "prediction": "CHURN" if prediction == 1 else "NON-CHURN",
        "probabilite": round(proba, 2),

        "reason": reason,
        "recommandation": recommandation if prediction == 1 else "Aucune recommandation requise"

    }




@app.post("/predict")
def predict(client: ClientFeatures):
    try:
        return predict_client(client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-csv")
def process_csv(file: UploadFile = File(...)):
    try:
        input_path = f"temp_{file.filename}"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        subprocess.run([
            "C:/Users/user/AppData/Local/Microsoft/WindowsApps/python3.11.exe",
            "C:/Users/user/Downloads/ChurnPrediction/churn_data_processing.py",
            input_path
        ], check=True)

        output_path = "processed_data.csv"
        if not os.path.exists(output_path):
            raise FileNotFoundError("Fichier preprocessé introuvable.")

        with open(output_path, 'rb') as f:
            files = {'file': ('processed_data.csv', f, 'text/csv')}
            response = requests.post("http://localhost:8090/clients/receive-csv", files=files)

        df = pd.read_csv(output_path)
        predictions_results = []

        for _, row in df.iterrows():
            try:
                client = ClientFeatures(
                    CLI_id=int(row["CLI_id"]),
                    NBENF=int(row["NBENF"]),
                    SEG=int(row["SEG"]),
                    nb_transactions=int(row["nb_transactions"]),
                    montant_total=float(row["montant_total"]),
                    montant_moyen=float(row["montant_moyen"]),
                    montant_max=float(row["montant_max"]),
                    montant_min=float(row["montant_min"]),
                    dernier_montant=float(row["dernier_montant"]),
                    nb_types_produits=int(row["nb_types_produits"]),
                    nb_libelles_produits=int(row["nb_libelles_produits"]),
                    age=int(row["age"]),
                    anciennete=int(row["anciennete"]),
                    SEXT=str(row["SEXT"])
                )
                result = predict_client(client)
                result["CLI_id"] = client.CLI_id
                predictions_results.append(result)

            except Exception as e:
                predictions_results.append({
                    "CLI_id": row.get("CLI_id", "unknown"),
                    "error": str(e)
                })

        return {
            "backend_response": response.text,
            "predictions": predictions_results
        }

    except subprocess.CalledProcessError as e:
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Erreur script preprocessing : {str(e)}")

    except Exception as e:
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")
