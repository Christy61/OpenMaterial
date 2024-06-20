from refractiveindex import RefractiveIndexMaterial
import mitsuba as mi
import pandas as pd
import os
from tqdm import tqdm

        
def get_example_objects(json_file) -> pd.DataFrame:
    """Returns a DataFrame of example objects to use for debugging."""
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)  
    return pd.read_json(json_file, orient="records")


if __name__ == '__main__':
    json_file = 'resource/data/conductor.json'
    print("start download...")
    mi.set_variant("scalar_rgb")
    os.makedirs('resource/data/ior_conductor', exist_ok=True)
    root = f'resource/data/ior_conductor'
    objects = get_example_objects(json_file)
    print(objects)
    names = objects['name'].to_list()
    shelfs = objects['shelf'].to_list()
    books = objects['book'].to_list()
    pages = objects['page'].to_list()
    print(names)
    wavelength_list = [i for i in range(380, 781, 5)]
    for i in tqdm(range(len(books))):
        material = RefractiveIndexMaterial(shelf=shelfs[i], book=books[i], page=pages[i])
        n = []
        k = []
        for wavelength_nm in wavelength_list:
            n.append(material.get_refractive_index(wavelength_nm))
            k.append(material.get_extinction_coefficient(wavelength_nm))
        filename_n = os.path.join(root, f'{names[i]}.eta.spd')
        filename_k = os.path.join(root, f'{names[i]}.k.spd')
        mi.spectrum_to_file(filename_n, wavelength_list, n)
        mi.spectrum_to_file(filename_k, wavelength_list, k)

    json_file = 'resource/data/dielectric.json'
    os.makedirs('resource/data/ior_dielectric', exist_ok=True)
    root = f'resource/data/ior_dielectric'
    objects = get_example_objects(json_file)
    print(objects)
    names = objects['name'].to_list()
    shelfs = objects['shelf'].to_list()
    books = objects['book'].to_list()
    pages = objects['page'].to_list()
    print(names)
    wavelength = 589.29
    for i in tqdm(range(len(books))):
        material = RefractiveIndexMaterial(shelf=shelfs[i], book=books[i], page=pages[i])
        nd = material.get_refractive_index(wavelength)
        filename_nd = os.path.join(root, f'{names[i]}.txt')
        text = f'{wavelength} {nd}'
        with open(filename_nd, 'w') as file:
            file.write(text)

    json_file = 'resource/data/plastic.json'
    os.makedirs('resource/data/ior_plastic', exist_ok=True)
    root = f'resource/data/ior_plastic'
    objects = get_example_objects(json_file)
    print(objects)
    names = objects['name'].to_list()
    shelfs = objects['shelf'].to_list()
    books = objects['book'].to_list()
    pages = objects['page'].to_list()
    print(names)
    wavelength = 589.29
    for i in tqdm(range(len(books))):
        material = RefractiveIndexMaterial(shelf=shelfs[i], book=books[i], page=pages[i])
        nd = material.get_refractive_index(wavelength)
        filename_nd = os.path.join(root, f'{names[i]}.txt')
        text = f'{wavelength} {nd}'
        with open(filename_nd, 'w') as file:
            file.write(text)
