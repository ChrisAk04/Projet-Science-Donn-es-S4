# ===============================
# PIPELINE COMPLET : Classification ou Régression Multimodale en 5 Classes
# (Apprentissage Amélioré & Visualisations)
# Ce code a été optimisé grace à ChatGPT
# ===============================

# === IMPORTATION DES LIBRAIRIES ===
# Bibliothèques système et manipulation de fichiers
import os, glob, json
# Manipulation de données et visualisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
import seaborn as sns

# Bibliothèques pour le Deep Learning (PyTorch & Transformers)
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
# Méthodes de prétraitement et d'évaluation scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error,
                             mean_squared_error, r2_score)

# === CHOIX DE LA TÂCHE ===
# On demande à l'utilisateur s'il veut faire de la classification ou de la régression
TASK = ''
while TASK not in ['classification', 'regression']:
    TASK = input("Veuillez choisir une tâche ('classification' ou 'regression') : ").strip().lower()

# === DÉFINITION DES CHEMINS D'ACCÈS AUX FICHIERS ===
BASE_DIR = "./images/"
MCPS_PATH = "./insurance.csv"
TEXT_PATH = "./mtsamples.csv"
CSV_META_PATH = BASE_DIR + "Data_Entry_2017.csv"

# === CHARGEMENT DES DONNÉES ===
# Chargement des transcriptions médicales et nettoyage
transcriptions = pd.read_csv(TEXT_PATH).dropna(subset=['transcription'])
transcriptions['keywords'] = transcriptions['keywords'].fillna('').str.lower()
transcriptions['medical_specialty'] = transcriptions['medical_specialty'].fillna('').str.lower()
transcriptions['description'] = transcriptions['description'].fillna('').str.lower()

# Chargement des données tabulaires sur les coûts médicaux (MCPS)
mcps = pd.read_csv(MCPS_PATH)

# Chargement des métadonnées des images (radiographies)
meta = pd.read_csv(CSV_META_PATH)

# === TRAITEMENT DES MÉTADONNÉES D'IMAGES ===
# On liste tous les chemins vers les images disponibles (format PNG)
all_image_paths = glob.glob(os.path.join(BASE_DIR, "images_*", "images", "*.png"), recursive=True)
# Création d'un dictionnaire : nom d'image → chemin complet
image_path_dict = {os.path.basename(p): p for p in all_image_paths}

# Sélection et nettoyage des colonnes pertinentes des métadonnées
meta = meta[['Image Index', 'Patient ID', 'Patient Age', 'Patient Gender', 'Finding Labels']]
meta['Patient Age'] = pd.to_numeric(meta['Patient Age'], errors='coerce')
meta['Patient Gender'] = meta['Patient Gender'].astype(str).str.strip().str.upper()
meta = meta.dropna(subset=['Patient Age', 'Patient Gender'])
meta = meta[(meta['Patient Age'] > 0) & (meta['Patient Age'] < 100)]
meta = meta[meta['Patient Gender'].isin(['M', 'F'])]
# Association des noms de fichiers aux chemins d’images réels
meta['full_path'] = meta['Image Index'].map(image_path_dict)
# Suppression des entrées sans image valide
meta = meta[meta['full_path'].notnull()].reset_index(drop=True)

# === PRÉTRAITEMENT DES DONNÉES TABULAIRES ===
# Nettoyage des champs catégoriels (sexe, tabagisme)
mcps = mcps.dropna(subset=['sex'])
mcps['sex'] = mcps['sex'].astype(str).str.strip().str.lower().map({'male': 'M', 'female': 'F'})
mcps['smoker'] = mcps['smoker'].fillna('no').str.lower()
mcps['raw_age'] = mcps['age'].astype(int)

# Colonnes quantitatives à normaliser
tabular_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
mcps[tabular_cols] = scaler.fit_transform(mcps[tabular_cols])
mcps[tabular_cols] = mcps[tabular_cols].astype(np.float32)

# Sauvegarde des statistiques de normalisation (utile en régression pour dénormaliser)
charge_mean = mcps['charges'].mean()
charge_std = mcps['charges'].std()

# === NORMALISATION POUR LA RÉGRESSION ===
if TASK == 'regression':
    mcps['charges'] = (mcps['charges'] - charge_mean) / charge_std

# === CRÉATION DES CLASSES POUR LA CLASSIFICATION ===
if TASK == 'classification':
    def create_five_cost_classes(mcps):
        """
        Crée 5 classes de coût basées sur des quantiles pour assurer un bon équilibre entre les classes.
        Affiche les seuils et la répartition.
        """
        # On crée les classes avec les quantiles
        mcps['cost_class'], bins = pd.qcut(mcps['charges'], q=5, labels=False, retbins=True, duplicates='drop')

        print("📊 Seuils de coupure utilisés :", bins.round(2).tolist())
        print("📊 Répartition des classes :")
        print(mcps['cost_class'].value_counts().sort_index())

        return mcps


    # Application de la fonction de classification sur le dataset
    mcps = create_five_cost_classes(mcps)


# === ASSOCIATION D'UNE IMAGE À CHAQUE INDIVIDU ===
def match_image(row, meta, max_tolerance=2):
    """
    Essaie de trouver une image correspondante pour un individu donné en fonction du sexe, de l'âge,
    du statut de fumeur et de l'IMC, avec une tolérance d'âge.
    """
    sex, age, bmi, smoker = row['sex'], int(row['age']), row['bmi'], row['smoker']
    for tol in range(max_tolerance + 1):
        candidates = meta[(meta['Patient Gender'] == sex) & (meta['Patient Age'] == age + tol)]
        if tol > 0:
            candidates = pd.concat(
                [candidates, meta[(meta['Patient Gender'] == sex) & (meta['Patient Age'] == age - tol)]])
        if smoker == 'yes':
            candidates = candidates[
                candidates['Finding Labels'].str.contains("infiltration|emphysema|copd|pneumonia", case=False,
                                                          na=False)]
        if bmi > 30:
            candidates = candidates[
                candidates['Finding Labels'].str.contains("cardiomegaly|effusion|atelectasis", case=False, na=False)]
        if not candidates.empty:
            return candidates.sample(1)['full_path'].values[0]
    return None


# Application de l'association image à chaque ligne du jeu de données
mcps['image_path'] = mcps.apply(lambda row: match_image(row, meta, 2), axis=1)
mcps = mcps[mcps['image_path'].notnull()].reset_index(drop=True)


# === ASSOCIATION D’UNE TRANSCRIPTION TEXTUELLE À CHAQUE INDIVIDU ===
def match_transcription(row, mtsamples):
    """
    Associe une transcription médicale pertinente en fonction de l'âge, du sexe,
    du nombre d'enfants, du statut tabagique et de l'IMC.
    """
    filters = []
    if row['smoker'] == 'yes':
        filters.append(mtsamples['keywords'].str.contains("smoke|lung|respiratory|cardio|copd"))
    if row['bmi'] > 30:
        filters.append(mtsamples['keywords'].str.contains("obese|weight|bariatric|diet|nutrition"))
    if row['raw_age'] < 18:
        filters.append(mtsamples['medical_specialty'].str.contains("pediatrics"))
    elif row['raw_age'] > 65:
        filters.append(mtsamples['medical_specialty'].str.contains("geriatrics|internal"))
    if row['children'] > 2:
        filters.append(mtsamples['keywords'].str.contains("family|pregnancy|childbirth|obstetric"))
    if row['sex'] == 'F':
        filters.append(mtsamples['keywords'].str.contains("gynecology|pregnancy|female health"))
    if row['sex'] == 'M':
        filters.append(~mtsamples['keywords'].str.contains("pregnancy|obstetric", na=False))

    if filters:
        combined = filters[0]
        for f in filters[1:]:
            combined &= f
        filtered = mtsamples[combined]
    else:
        filtered = mtsamples

    return filtered.sample(1)['transcription'].values[0] if not filtered.empty else \
        mtsamples.sample(1)['transcription'].values[0]


# Application à chaque individu
mcps['text'] = mcps.apply(lambda row: match_transcription(row, transcriptions), axis=1)

# === SÉPARATION TRAIN / TEST ===
# On sépare les données pour l'entraînement et le test, en stratifiant si classification
label_column = 'cost_class' if TASK == 'classification' else 'charges'
train_df, test_df = train_test_split(mcps, test_size=0.2,
                                     stratify=mcps['cost_class'] if TASK == 'classification' else None, random_state=42)

# === VISUALISATIONS AVANT ENTRAÎNEMENT ===
# Distribution des classes ou des charges
if TASK == 'classification':
    sns.countplot(x='cost_class', data=mcps)
    plt.title("Class Distribution")
    plt.savefig("class_distribution.png")
else:
    plt.hist(mcps['charges'], bins=50)
    plt.title("Charges Distribution")
    plt.xlabel("Charges")
    plt.ylabel("Count")
    plt.savefig("charges_histogram.png")

# Nuage de mots basé sur le corpus des transcriptions
wordcloud = WordCloud(width=800, height=400).generate(' '.join(mcps['text']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud")
plt.savefig("text_wordcloud.png")

# Affichage de 5 exemples d’images de radiographie
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axs[i].imshow(Image.open(mcps['image_path'].iloc[i]).convert('L'), cmap='gray')
    axs[i].axis('off')
plt.suptitle("Sample Chest X-rays")
plt.savefig("sample_images.png")

# === TOKENIZER BERT POUR LE TEXTE ===
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# === CLASSE DATASET MULTIMODAL ===
class MultimodalDataset(Dataset):
    def __init__(self, df, tabular_cols, tokenizer, transform=None):
        self.df = df.reset_index(drop=True)
        self.tabular_cols = tabular_cols
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tabular = torch.tensor(row[self.tabular_cols].values.astype(np.float32), dtype=torch.float32)
        tokens = self.tokenizer(row['text'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        text_data = {k: v.squeeze(0) for k, v in tokens.items()}
        image = Image.open(row['image_path']).convert("RGB") if os.path.exists(row['image_path']) else torch.zeros(
            (3, 224, 224))
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row[label_column], dtype=torch.float if TASK == 'regression' else torch.long)
        return {'tabular': tabular, 'text': text_data, 'image': image, 'target': label}


# === ENCODAGE DES MODALITÉS ===

# Encodeur des données tabulaires via MLP simple
class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


# Encodeur du texte à l’aide de BERT pré-entraîné
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output


# Encodeur d’images basé sur ResNet18 pré-entraîné
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        base.fc = nn.Identity()  # On supprime la couche finale de classification
        self.cnn = base

    def forward(self, x):
        return self.cnn(x)


# Fusionneur multimodal combinant les trois encodeurs
class MultimodalModel(nn.Module):
    def __init__(self, tabular_dim, output_dim):
        super().__init__()
        self.tab_enc = TabularEncoder(tabular_dim)
        self.text_enc = TextEncoder()
        self.img_enc = ImageEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(64 + 768 + 512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, tab, text, img):
        z1 = self.tab_enc(tab)
        z2 = self.text_enc(text['input_ids'], text['attention_mask'])
        z3 = self.img_enc(img)
        return self.classifier(torch.cat([z1, z2, z3], dim=1))


# === TRANSFORMATIONS POUR LES IMAGES ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Création des datasets et dataloaders
train_ds = MultimodalDataset(train_df, tabular_cols, tokenizer, transform)
test_ds = MultimodalDataset(test_df, tabular_cols, tokenizer, transform)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

# === INITIALISATION DU MODÈLE ===
num_classes = 5 if TASK == 'classification' else 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalModel(tabular_dim=len(tabular_cols), output_dim=num_classes).to(device)

# Fonction de perte selon la tâche choisie
criterion = nn.CrossEntropyLoss() if TASK == 'classification' else nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
epochs = 20

# === BOUCLE D'ENTRAÎNEMENT ===
train_losses = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        tab = batch['tabular'].to(device)
        text = {k: v.to(device) for k, v in batch['text'].items()}
        img = batch['image'].to(device)
        tgt = batch['target'].to(device)

        optimizer.zero_grad()
        out = model(tab, text, img)
        if TASK == 'regression':
            out = out.squeeze(1)
        loss = criterion(out, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# Sauvegarde de la courbe de perte d'entraînement
plt.figure()
plt.plot(train_losses)
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_curve.png")

# === ÉVALUATION DU MODÈLE ===
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        tab = batch['tabular'].to(device)
        text = {k: v.to(device) for k, v in batch['text'].items()}
        img = batch['image'].to(device)
        tgt = batch['target'].to(device)
        out = model(tab, text, img)
        if TASK == 'regression':
            out = out.squeeze(1)
            preds = out
        else:
            preds = torch.argmax(out, dim=1)
        y_true.extend(tgt.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

# === VISUALISATIONS POUR LA CLASSIFICATION ===
if TASK == 'classification':
    from sklearn.utils.multiclass import unique_labels

    labels = unique_labels(y_true, y_pred)
    label_names = ["Very Low", "Low", "Medium", "High", "Very High"]
    display_names = [label_names[i] for i in labels]

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    # Rapport de classification en format JSON
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, target_names=display_names)
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Barres : précision, rappel et F1-score
    df_report = pd.DataFrame(report).T.loc[display_names][['precision', 'recall', 'f1-score']]
    plt.figure()
    df_report.plot(kind='bar')
    plt.title("Precision, Recall, F1-Score per Class")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("classification_metrics_per_class.png")

    # t-SNE : Visualisation des embeddings appris
    from sklearn.manifold import TSNE

    embeddings, tsne_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            tab = batch['tabular'].to(device)
            text = {k: v.to(device) for k, v in batch['text'].items()}
            img = batch['image'].to(device)
            z1 = model.tab_enc(tab)
            z2 = model.text_enc(text['input_ids'], text['attention_mask'])
            z3 = model.img_enc(img)
            z = torch.cat([z1, z2, z3], dim=1)
            embeddings.append(z.cpu().numpy())
            tsne_labels.extend(batch['target'].cpu().numpy())

    embeddings = np.concatenate(embeddings)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)
    plt.figure()
    sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=tsne_labels, palette='tab10', legend='full')
    plt.title("t-SNE of Learned Embeddings")
    plt.savefig("tsne_embeddings.png")

# === VISUALISATIONS POUR LA RÉGRESSION ===
else:
    # Dénormalisation des prédictions et cibles
    y_true = np.array(y_true) * charge_std + charge_mean
    y_pred = np.array(y_pred) * charge_std + charge_mean

    # Histogramme des erreurs absolues
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    plt.hist(errors, bins=40)
    plt.title("Prediction Errors")
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.savefig("prediction_error_histogram.png")

    # Nuage de points : vrai vs prédit
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.title("True vs Predicted Charges")
    plt.xlabel("True Charges")
    plt.ylabel("Predicted Charges")
    plt.savefig("scatter_true_vs_pred.png")

    # Résidus vs valeurs prédites
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted Charges")
    plt.ylabel("Residuals")
    plt.savefig("residuals_vs_predicted.png")

    # Densité : distribution des valeurs vraies et prédites
    plt.figure()
    sns.kdeplot(y_true, label="True Charges", shade=True)
    sns.kdeplot(y_pred, label="Predicted Charges", shade=True)
    plt.title("Distribution of True vs Predicted Charges")
    plt.xlabel("Charges")
    plt.legend()
    plt.savefig("distribution_true_vs_predicted.png")

    # Boîte à moustaches : erreur par tranche d’âge
    df_eval = pd.DataFrame({
        'error': np.abs(np.array(y_true) - np.array(y_pred)),
        'age': test_df['raw_age'].values,
        'class': test_df['cost_class'] if 'cost_class' in test_df.columns else pd.cut(test_df['charges'], bins=5)
    })
    df_eval['age_group'] = pd.cut(df_eval['age'], bins=[0, 25, 40, 60, 100], labels=["<25", "25-40", "40-60", "60+"])
    plt.figure()
    sns.boxplot(x='age_group', y='error', data=df_eval)
    plt.title("Prediction Error by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Absolute Error")
    plt.savefig("error_by_age_group.png")

    # Courbes triées : valeurs réelles vs prédites
    sorted_idx = np.argsort(y_true)
    plt.figure()
    plt.plot(np.array(y_true)[sorted_idx], label='True')
    plt.plot(np.array(y_pred)[sorted_idx], label='Predicted')
    plt.title("True vs Predicted Charges (Sorted)")
    plt.legend()
    plt.savefig("true_vs_predicted_sorted.png")

    # Calcul et export des métriques classiques
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),
        'R2': r2_score(y_true, y_pred)
    }
    with open("regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
