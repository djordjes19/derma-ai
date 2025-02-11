# Projekat za Detekciju Melanoma

Ovaj projekat se bavi klasifikacijom dermatoskopskih slika melanoma koristeći tehnike mašinskog učenja. Korišćenjem skupa podataka sa slikama i oznakama iz [ISIC (International Skin Imaging Collaboration)](https://www.isic-archive.com/), cilj je da se izgrade modeli koji mogu efikasno detektovati maligne lezije kože. Projekat uključuje analizu podataka, treniranje modela, evaluaciju i implementaciju u stvarnom vremenu.

## Struktura Projekta

```plaintext
📂 projekat-melanom/
├── data/                 # Slikovni podaci (neće biti pohranjeni u git repozitorijumu)
├── ground_truth.csv      # CSV fajl sa oznakama (Training Ground Truth)
├── analiza.ipynb         # Jupyter notebook sa analizom podataka i treniranjem modela
