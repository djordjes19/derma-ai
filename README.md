# Projekat za Detekciju Melanoma

Ovaj projekat se bavi klasifikacijom dermatoskopskih slika melanoma koristeći tehnike mašinskog učenja. Korišćenjem skupa podataka sa slikama i oznakama iz [ISIC (International Skin Imaging Collaboration)](https://www.isic-archive.com/), cilj je da se izgrade modeli koji mogu efikasno detektovati maligne lezije kože. Projekat uključuje analizu podataka, treniranje modela, evaluaciju i implementaciju u stvarnom vremenu.

# Koraci
1. Naci odgovarajuci model za anotaciju boje koze.
2. Dodati zasebni feature u dataset sa tom opcijom
3. Odraditi vizuelnu inspekciju rasprostranjenosti atributa.
4. Odraditi balansiranje skupa.


# Treniranje pokrecete na komandu
python src/models/train.py -- config config.yaml

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   


