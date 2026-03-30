# 📞 Prédiction du Churn Client — Telco

Projet de machine learning de bout en bout sur le dataset **Telco Customer Churn**.

---

## 📌 Problème traité

Prédire si un client d'un opérateur télécom va **résilier son abonnement** (churn).
C'est un problème de **classification binaire** déséquilibrée (~26% de churners).

**Enjeu métier** : retenir un client coûte 5× moins cher qu'en acquérir un nouveau.
Un bon modèle permet de cibler des actions de rétention proactives.

---

## 📁 Dataset

| Propriété | Valeur |
|---|---|
| Source | IBM Sample Data / [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Lignes | 7 043 clients |
| Features | 20 (mix numérique + catégoriel) |
| Cible | `Churn` — Yes (a résilié) / No (est resté) |
| Déséquilibre | ~74% Non-churn / ~26% Churn |

**Variables clés** : tenure, MonthlyCharges, TotalCharges, Contract,
InternetService, PaymentMethod, TechSupport, etc.

---

## 🛠️ Pipeline

```
Dataset → Nettoyage → Encodage OHE → Split stratifié 80/20
       → Normalisation → SMOTE → Modèles → Évaluation → Interface
```

**Points spécifiques à la classification :**
- Encodage One-Hot des 16 variables catégorielles
- Split **stratifié** (préserve le ratio churn dans train/test)
- **SMOTE** pour rééquilibrer les classes (train uniquement)
- Métriques adaptées : F1-score, AUC-ROC, matrice de confusion

**Partie non supervisée (bonus) :**
- PCA 2D pour visualiser la séparation churn/non-churn
- K-Means (k=4) pour identifier les segments clients à risque

---

## 📊 Résultats

| Modèle | Accuracy | F1-score | AUC-ROC |
|---|---|---|---|
| Régression Logistique | ~0.77 | ~0.60 | ~0.84 |
| Random Forest | ~0.80 | ~0.64 | ~0.87 |
| Gradient Boosting | ~0.81 | **~0.65** | **~0.88** |

🏆 **Meilleur modèle : Gradient Boosting** (AUC-ROC ≈ 0.88)

---

## ▶️ Lancer le projet

### 1. Télécharger le dataset

Depuis Kaggle : https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Placer `WA_Fn-UseC_-Telco-Customer-Churn.csv` dans le dossier du notebook.

> Sans le fichier, le notebook génère automatiquement un dataset synthétique réaliste.

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer Jupyter

```bash
jupyter notebook notebook.ipynb
```

### 4. Exécuter toutes les cellules

`Kernel` → `Restart & Run All`

La dernière cellule affiche une interface interactive pour tester le churn
d'un client en ajustant son profil (ancienneté, contrat, charges...).

---

## 📦 Structure du projet

```
churn_prediction/
├── notebook.ipynb                          ← Pipeline complet + interface
├── requirements.txt                        ← Dépendances Python
├── README.md                               ← Ce fichier
├── generate_report.py                      ← Script pour générer rapport.pdf
├── rapport.pdf                             ← Rapport PDF (7-10 pages)
└── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← Dataset (à télécharger sur Kaggle)
```

---

## 👥 Auteurs

- Étudiant  — Omar Boussiline

