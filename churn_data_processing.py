import pandas as pd
import sys
import os
from functools import reduce

# Fonction principale de prétraitement
def preprocess_data(file_path):
    # ─── Chargement du fichier ──────────────────────────────────────────────
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, low_memory=False)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
        if len(df.columns) == 1 and ',' in df.columns[0]:
            print("Colonnes fusionnées détectées, tentative de lecture comme CSV")
            df = pd.read_csv(file_path, low_memory=False)
    else:
        raise ValueError("Format non supporté")

    # ─── Conversion des dates ───────────────────────────────────────────────
    df["DCO"] = pd.to_datetime(df["DCO"], errors='coerce')
    df["DOU"] = pd.to_datetime(df["DOU"], errors='coerce')
    df["DNA"] = pd.to_datetime(df["DNA"], errors='coerce')

    # ─── Nettoyage initial ──────────────────────────────────────────────────
    df.dropna(inplace=True)

    # ─── Table statique client ──────────────────────────────────────────────
    df_static = df.groupby("CLI").agg({
        "SEXT": "first",
        "NBENF": "first",
        "SEG": "first",
        "DNA": "first",
        "AGE_NCP_HASH": "first"
    }).reset_index()

    # ─── Nombre de transactions par client ──────────────────────────────────
    df_nb_txn = df.groupby("CLI").size().reset_index(name="nb_transactions")

    # ─── Statistiques sur les montants ──────────────────────────────────────
    df_montant = df.groupby("CLI")["MON"].agg(["sum", "mean", "max", "min"]).reset_index()
    df_montant.columns = ["CLI", "montant_total", "montant_moyen", "montant_max", "montant_min"]

    # ─── Dernière transaction par client ────────────────────────────────────
    df_last = df.sort_values("DCO").groupby("CLI").last().reset_index()
    df_last_txn = df_last[["CLI", "DCO", "BH_LIB", "MON"]].rename(columns={
        "DCO": "date_derniere_txn",
        "BH_LIB": "dernier_type_op",
        "MON": "dernier_montant"
    })

    # ─── Diversité des produits ─────────────────────────────────────────────
    df_produits = df.groupby("CLI").agg({
        "CPRO": pd.Series.nunique,
        "LIB": pd.Series.nunique
    }).reset_index().rename(columns={
        "CPRO": "nb_types_produits",
        "LIB": "nb_libelles_produits"
    })

    # ─── Fusion de toutes les tables ────────────────────────────────────────
    dfs = [df_static, df_nb_txn, df_montant, df_last_txn, df_produits]
    df_clients_final = reduce(lambda left, right: pd.merge(left, right, on="CLI", how="left"), dfs)

    # ─── Calcul de l’ancienneté et de l’âge ─────────────────────────────────
    df_clients_final["anciennete"] = (pd.Timestamp.now() - df["DOU"]).dt.days // 365
    df_clients_final["age"] = (pd.Timestamp.now() - df_clients_final["DNA"]).dt.days // 365

    # ─── Mapping CLI vers identifiants entiers ──────────────────────────────
    cli_mapping = {cli: idx+1 for idx, cli in enumerate(df_clients_final['CLI'].unique())}
    df_clients_final['CLI_id'] = df_clients_final['CLI'].map(cli_mapping)

    # ─── Réorganisation des colonnes ────────────────────────────────────────
    col_to_move = 'CLI_id'
    cols = [col_to_move] + [col for col in df_clients_final.columns if col != col_to_move]
    df_clients_final = df_clients_final[cols]

    # ─── Suppression des colonnes inutiles ──────────────────────────────────
    df_clients_final = df_clients_final.drop(columns=["AGE_NCP_HASH", "CLI", "DNA", "date_derniere_txn"])

    # ─── Conversion explicite des types ─────────────────────────────────────
    type_map = {
        "age": int,
        "dernier_montant": float,
        "nb_transactions": int,
        "nb_types_produits": int,
    }

    for col, dtype in type_map.items():
        if col in df_clients_final.columns:
            df_clients_final[col] = df_clients_final[col].astype(dtype)
        else:
            print(f"Colonne '{col}' absente, sautée.")

    # ─── Affichage complet pour debug éventuel ──────────────────────────────
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', 100)
    return df_clients_final

# ─── Exécution principale ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = os.path.join(os.path.dirname(input_path), "processed_data.csv")

    try:
        df_processed = preprocess_data(input_path)
        df_processed.to_csv(output_path, index=False)
        print(f"Prétraitement terminé. Fichier enregistré : {output_path}")
    except Exception as e:
        print(f"Erreur lors du traitement des données : {str(e)}")
        sys.exit(1)
