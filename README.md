# Reconnaissance et analyse automatique de panneaux de signalisation

Projet de Deep Learning (Master, semestre 1, 2026-2027) : un système qui **détecte** les panneaux de signalisation dans une photo ou une vidéo de rue avec YOLO, puis **reconnaît** chaque panneau avec un réseau de neurones convolutif (CNN) entraîné avec Keras.

Équipe : Efe Tatar, Marius Brosset, Nathan Signoud (respectivement groupes A, A et B) 

## Résultats

| Modèle | Tâche | Résultat sur le test |
|---|---|---|
| MLP (référence) | classification des panneaux découpés | accuracy 87,8 % |
| CNN (partie 3) | classification des panneaux découpés | accuracy 91,1 %, F1 macro 0,82 |
| **CNN final** (dropout 0,3) | classification des panneaux découpés | **accuracy 93,3 %, F1 macro 0,87** |
| YOLOv8 nano fine-tuné | détection + classification (48 classes) | mAP50 0,79, mAP50-95 0,70 |
| **YOLO + CNN final** | photos complètes | 93,1 % des panneaux trouvés ; **94,7 %** bien classés par le CNN, contre 91,9 % par YOLO seul |

Le pipeline en deux étapes est le plus performant : YOLO localise les panneaux, et le CNN, qui reçoit chaque panneau découpé et agrandi, les classe mieux que la branche de classification de YOLO.

## Jeu de données

[YOLO to COCO](https://universe.roboflow.com/yolo-to-coco-ca2en/yolo-to-coco-kpv56) (Roboflow Universe) : 3 053 photos de scènes routières et 5 983 panneaux annotés par une boîte et une classe, répartis en 54 classes (interdiction, obligation, danger, priorité, indication).

Le notebook retire les 6 classes qui ont moins de 15 exemples (48 classes restantes), puis refait la séparation train / validation / test (70 / 15 / 15) **par photo** et en s'assurant que chaque classe est présente dans les trois ensembles.

## Lancer le notebook

Le notebook [projet.ipynb](projet.ipynb) est fait pour **Google Colab**.

1. **Copier le dataset** (*fork*) dans votre propre espace Roboflow depuis la page du dataset, puis récupérer votre clé API dans les paramètres Roboflow.
2. **Ajouter la clé dans Colab :** icône de clé dans la barre de gauche, nouveau secret nommé `ROBOFLOW_API_KEY`, accès au notebook activé. Dans la cellule de téléchargement, remplacez aussi l'espace de travail et le nom du projet (`efe-tatar`, `yolo-to-coco-traffic-signs-pmo2q`) par les vôtres.
3. **Choisir un GPU :** *Runtime → Change runtime type → T4 GPU*.
4. **Exécuter tout le notebook dans l'ordre** (*Runtime → Run all*). Compter environ une heure : une vingtaine de minutes pour les expériences de la partie 6 et 20 à 25 minutes pour l'entraînement de YOLO.

Les résultats sont reproductibles : les graines aléatoires de Python, NumPy et TensorFlow sont fixées, et TensorFlow est mis en mode déterministe.

## Contenu du notebook

| Partie | Contenu |
|---|---|
| 1. Données | description du dataset, statistiques (classes, tailles, format, déséquilibre), seuil sur les classes rares, nouvelle séparation, découpage, redimensionnement en 32 × 32, normalisation, encodage des labels |
| 2. Modèle de référence | MLP : Flatten → Dense(128) → Dense(48, softmax) |
| 3. CNN | deux blocs convolution + ReLU + max pooling, puis Dense ; tableau des dimensions et des paramètres |
| 4. Entraînement | justification de la perte, de l'optimiseur, du batch size et des époques ; courbes de loss et d'accuracy, surapprentissage |
| 5. Évaluation | matrice de confusion, précision / rappel / F1 par classe, confusions fréquentes, exemples d'erreurs, effet de la taille et de la luminosité |
| 6. Expériences | exploration de 7 hyperparamètres (un seul à la fois), combinaison des meilleures valeurs, data augmentation sur le MLP et le CNN, choix du modèle final sur le F1 macro de validation |
| 7. Enrichissement | data augmentation, régularisation, transfer learning |
| 8. Détection | YOLOv8 nano fine-tuné, pipeline YOLO → découpage → CNN, comparaison de la classification de YOLO et du CNN sur les mêmes boîtes |
| 9. Vidéo | analyse image par image et création d'une vidéo MP4 annotée, dans le dashboard |

## Dashboard

La dernière cellule lance un dashboard **Gradio** et affiche un lien public (valable 72 heures). Il contient quatre onglets :

- **Jeu de test :** photos complètes (YOLO + classifieur) ou panneaux seuls, filtrés par classe, avec la vraie classe à côté de la prédiction.
- **Image :** une image envoyée ou un lien direct vers un fichier `.jpg` / `.png`.
- **Vidéo :** une vidéo envoyée ou un lien direct vers un fichier `.mp4` ; la vidéo annotée est enregistrée dans `/content/videos/`.
- **Comparaison :** courbes de loss et d'accuracy, métriques de validation et de test, F1 par classe pour les modèles choisis, et courbes d'entraînement de YOLO.

Un menu permet de choisir le classifieur : MLP, CNN de la partie 3, chacune des variantes de la partie 6, ou YOLO seul.

Le modèle a appris des panneaux **européens** : pour tester une vidéo, prenez une rue européenne en plein jour (les vidéos libres de droits de [Pexels](https://www.pexels.com/videos/) conviennent). Un lien de page Pexels se transforme en lien direct avec `https://www.pexels.com/download/video/<numéro de la vidéo>/`.

## Remarques techniques

- **TensorFlow et PyTorch ne partagent pas le GPU dans le même processus** (le kernel plante). YOLO (PyTorch) est donc entraîné dans un processus séparé avec la commande `yolo`, puis utilisé sur le CPU dans le notebook, ce qui suffit pour les prédictions.
- **Tout est perdu quand la session Colab s'arrête** (modèles, dataset, vidéos) : il faut alors réexécuter le notebook.
- Les secrets Colab ne sont accessibles que depuis l'interface de Colab. Si le notebook est exécuté depuis VS Code sur un runtime Colab, remplacez `userdata.get("ROBOFLOW_API_KEY")` par `getpass("Roboflow API key: ")` (module `getpass`), et n'écrivez jamais la clé dans le notebook.
