# Faster RCNN MMdtection Tutoriel 


## Format du jeu de données 
Le format de  dataset  utilisé  en entraînement  et aussi en validation est le format COCO. Puisque nos données sont en format  YOLO, une conversion sera nécessaire. Pour cela, on utilise le programme qui existe  sur le  lien  GIT  ci-dessous permettant de convertir notre jeu de données du format  YOLO  vers le format COCO. On y trouvera un guide d'utilisation.

[YOLO to COCO format converter](https://github.com/Taeyoung96/Yolo-to-COCO-format-converter)

Après avoir généré le format convenable, on organise nos données de la manière suivante :
<p align="center">
    <img src="data_arbre.PNG" width="200" height="250"  />
</p>

## Configuration
Dans le  framework  MMdetection  il y a plusieurs modèles dont chacun a  sa  propre configuration. Pour notre cas, on utilise la configuration du  faster  RCNN  qui existe dans le `emr_fpn_v2.py`.  
Afin d'adapter la  configuration avec un nouveau jeu de données, il faut modifier les éléments ci-dessus :
 - La liste des classes : il faut mettre à jour les noms des classes dans la liste dédiée à ça dans le fichier de configuration et aussi dans le fichier source du DataLoader du format COCO (`coco.py`).
 - Le nombre de classes : il faut modifier le nombre de  classe dans la configuration du modèle (dans le dictionnaire "bbox_head").
 - Équilibrage des données : si le jeu de données n'est pas  équilibré, on peut limiter l'effet négatif de ceci en introduisant des coefficients de pondération dans la  loss à travers la clé "class_weight" du dictionnaire "bbox_head".
 - Le chemin du jeu de données : il faut préciser le chemin du dossier d'images  et  celui du fichier d'annotations json  dans le dictionnaire "data".
 - Le chemin racine du jeu de données "data_root".
 - Chemin du modèle pré-entraîné : les modèles pré-entraînés peuvent être  téléchargé  depuis le lien [pretrained model](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn). Après le  téléchargement, il faut préciser son chemin avec la variable "load_from".
 - Chemin du dossier des résultats : il faut préciser le chemin du dossier dans lequel ils seront enregistrés les résultats de l'entraînement  (variable "work_dir").
 - La normalisation des données : il faut calculer la moyenne et l'écart type  du jeu de données d'entraînement, puis il faut les mettre à jour à travers la variable "img_norm_cfg".

Pour plus de détails sur le réglage des hyperparamètres  (nombre d'Epoch, taille de  batch, augmentation  des données etc.) dans les fichiers de configuration, veuillez lire le tutoriel de la bibliothèque  MMdetection dans le lien suivant : [Tutoriel MMdetection](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html).

## Commandes

Pour exécuter un script, il faut passer par la ligne de commande. Les exemples de commandes ci-dessous montrent comment lancer l'entraînement, évaluer les résultats et tester le modèle entraîné.
Lancer l'entrainement :

    python tools/train.py config_path.py 
Avec `config_path.py` est le fichier de configuration (`emr_fpn_v2.py` dans notre cas).

Afficher les courbes d'erreur :

    python tools/analysis_tools/my_analyze_logs.py plot_curve log_file_path.json --keys loss_cls loss_bbox --datatype val --legend loss_cls loss_bbox --out out_file_path.jpg
Afficher les courbes de précision :

    python tools/analysis_tools/my_analyze_logs.py plot_curve log_file_path.json --keys bbox_mAP_50 bbox_mAP --legend bbox_mAP_50  bbox_mAP --out out_file_path.jpg
Tester un modèle entrainé :
 

    python tools/test.py \
    config_path.py \
    work_dir/model.pth \
    --out result_path.pkl \
    --show-dir result_images_dir \
    --eval mAP \
    --show-score-thr 0.4
Afficher la matrice de confusion :

    python tools/conf.py \
    config_path.py \
    work_dir/model.pth \
    --show-dir result_images_dir \
    --work-dir work_dir \
    --show-score-thr 0.4