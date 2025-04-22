import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_attribute(df, attribute, figsize=(15, 8), title=None) :
    """
    Plotuje i bar chart i pie chart za rasprostranjenost atributa.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame koji sadrži podatke
    attribute : str
        Naziv kolone za koju se plotuje rasprostranjenost
    figsize : tuple, default=(15, 8)
        Veličina figure
    title : str, optional
        Naslov grafikona, ako nije navedeno koristiće se naziv atributa
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    value_counts = df[attribute].value_counts().sort_values(ascending=False)

    # Bar chart
    ax1.bar(value_counts.index, value_counts.values)
    ax1.set_ylabel('Broj uzoraka')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(value_counts.index, rotation=45, ha='right')

    # Pie chart
    percentages = 100 * value_counts / value_counts.sum()
    labels = [f"{idx} ({perc:.1f}%)" for idx, perc in zip(value_counts.index, percentages)]
    ax2.pie(value_counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')

    if title is None:
        title = f"Rasprostranjenost atributa: {attribute}"

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_skin_tone_distribution( df):
    # Učitavanje CSV fajla
    # Definišemo nazive kolona na osnovu primera
    # column_names = ['image_id', 'patient_id', 'gender', 'age', 'body_site',
    #                 'diagnosis_confirmation_type', 'diagnosis', 'melanoma', 'skin_tone']
    #
    # # Čitanje CSV fajla
    # df = pd.read_csv(csv_file, names=column_names)
    #df = df[df['melanoma'] == '1']
    # Grupisanje podataka po vrednosti boje kože
    skin_tone_counts = df['skin_tone'].value_counts().sort_index()

    # Kreiranje grafa
    plt.figure(figsize=(12, 6))

    # Kreiramo bar plot gde je svaki bar obojen odgovarajućom bojom
    bars = plt.bar(range(len(skin_tone_counts)), skin_tone_counts.values)

    # Postavljanje boje svakog bara prema heksadecimalnoj vrednosti
    for i, (hex_color, count) in enumerate(skin_tone_counts.items()):
        bars[i].set_color(hex_color)
        # Dodajemo tekst sa brojem podataka i heks kodom
        plt.text(i, count + 0.5, f"{count}\n{hex_color}",
                 ha='center', va='bottom', fontsize=8)

    # Podešavanje ose x da prikazuje heksadecimalne kodove
    plt.xticks(range(len(skin_tone_counts)), skin_tone_counts.index, rotation=45)

    # Dodavanje naslova i oznaka osa
    plt.title('Distribucija podataka po boji kože')
    plt.xlabel('Heksadecimalni kod boje kože')
    plt.ylabel('Broj uzoraka')

    plt.tight_layout()
    plt.show()

    # Dodatno, možemo prikazati i pie chart za vizuelni prikaz proporcija
    plt.figure(figsize=(10, 8))
    plt.pie(skin_tone_counts.values, labels=skin_tone_counts.index,
            colors=skin_tone_counts.index, autopct='%1.1f%%')
    plt.title('Proporcije podataka po boji kože')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()