import cv2
import numpy as np
import datetime
import matplotlib.cm as mpl_cm  # pre vlastnú farebnú mapu

def apply_custom_colormap(gray_img):
    """
    Aplikuje vlastnú farebnú mapu (viridis z Matplotlib) na normalizovaný grayscale obrázok.
    """
    # Normalizácia do intervalu 0-1
    normed = gray_img.astype(np.float32) / 255.0
    # Získanie colormap viridis (návrat RGBA v intervale 0-1)
    colormap = mpl_cm.get_cmap('viridis')
    colored = colormap(normed)
    # Odstránenie alfa kanála, prevod na rozsah 0-255 a premena na uint8
    colored = (colored[:, :, :3] * 255).astype(np.uint8)
    # Prevod z RGB na BGR (OpenCV používa BGR)
    colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    return colored

def make_composite(images, cols):
    """
    Zostaví zo zoznamu obrázkov kompozitný obrázok v mriežke so zadaným počtom stĺpcov.
    Ak posledný riadok nie je plný, doplní prázdne (čierne) obrázky.
    """
    rows = []
    for i in range(0, len(images), cols):
        row_imgs = images[i:i+cols]
        if len(row_imgs) < cols:
            h, w, ch = row_imgs[0].shape
            blank = np.zeros((h, w, ch), dtype=np.uint8)
            row_imgs.extend([blank] * (cols - len(row_imgs)))
        row = cv2.hconcat(row_imgs)
        rows.append(row)
    composite = cv2.vconcat(rows)
    return composite

def main():
    # Inicializácia kamery – v tomto prípade externá kamera s indexom 1
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Chyba: Kamera sa nepodarilo otvoriť.")
        return

    # Pokusíme sa nastaviť najvyššie možné rozlíšenie (tu 1920x1080)
    desired_width = 1920
    desired_height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    # Načítame prvý snímok pre získanie aktuálnych rozmerov
    ret, frame = cap.read()
    if not ret:
        print("Chyba: Nepodarilo sa načítať snímok.")
        return
    height, width = frame.shape[:2]
    print("Aktuálne rozlíšenie:", width, "x", height)

    # Akumulátor pre zachytenie intenzity pohybu
    accumulator = np.zeros((height, width), dtype=np.float32)

    # Inicializácia background subtractoru na extrakciu pohybu
    # (ak chceš ešte väčšiu citlivosť, môžeš znížiť varThreshold napr. na 10)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    print("Beží snímanie – stlač ESC pre ukončenie a export heatmapy.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Koniec streamu alebo chyba snímania.")
            break

        # Detekcia pohybu pomocou background subtractoru
        fgMask = backSub.apply(frame)
        # Zníženie prahovej hodnoty na 150 pre zvýšenie citlivosti
        _, fgMask = cv2.threshold(fgMask, 150, 255, cv2.THRESH_BINARY)
        # Zvýšenie veľkosti kernelu – použijeme väčší kernel (5x5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Aplikácia dilatácie pre "rozšírenie" detekovaných oblastí
        fgMask = cv2.dilate(fgMask, kernel, iterations=1)
        # Morfologická operácia otvárania pre odstránenie šumu
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

        # Akumulácia: pripočítavame detekovaný pohyb (masku prevedenú na interval 0-1)
        accumulator += fgMask.astype(np.float32) / 255.0

        # Pre vizualizáciu počas behu – použijeme pôvodný variant (COLORMAP_JET)
        norm_acc = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_vis = cv2.applyColorMap(norm_acc, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap_vis, 0.4, 0)

        cv2.imshow("Heatmap Overlay", overlay)
        cv2.imshow("Maska pohybu", fgMask)

        # Ukončenie po stlačení ESC (ASCII 27)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Finalizácia: normalizácia akumulátora do rozsahu 0-255
    norm_acc = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Varianty heatmapy:

    # 1. Pôvodná heatmapa (použitá farebná mapa JET)
    heatmap_original = cv2.applyColorMap(norm_acc, cv2.COLORMAP_JET)

    # 2. Variant s farebnou mapou INFERNO
    heatmap_inferno = cv2.applyColorMap(norm_acc, cv2.COLORMAP_INFERNO)

    # 3. Variant s farebnou mapou PLASMA
    heatmap_plasma = cv2.applyColorMap(norm_acc, cv2.COLORMAP_PLASMA)

    # 4. Variant s farebnou mapou HOT
    heatmap_hot = cv2.applyColorMap(norm_acc, cv2.COLORMAP_HOT)

    # 5. Variant s INFERNO a Gaussovským rozostrením
    heatmap_blur = cv2.GaussianBlur(heatmap_inferno, (11, 11), 0)

    # 6. Variant s vlastnou farebnou mapou (Viridis z Matplotlib)
    heatmap_custom = apply_custom_colormap(norm_acc)

    # 7. Variant s detekciou okrajov – použijeme Canny na normalizovaný akumulátor
    edges = cv2.Canny(norm_acc, 50, 150)
    heatmap_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Vytvorenie časovej pečiatky pre názvy súborov
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Uloženie jednotlivých variantov s najvyššou možnou kvalitou (PNG, kompresia = 0)
    cv2.imwrite(f"heatmap_original_{timestamp}.png", heatmap_original, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"heatmap_inferno_{timestamp}.png", heatmap_inferno, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"heatmap_plasma_{timestamp}.png", heatmap_plasma, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"heatmap_hot_{timestamp}.png", heatmap_hot, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"heatmap_blur_{timestamp}.png", heatmap_blur, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"heatmap_custom_{timestamp}.png", heatmap_custom, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"heatmap_edges_{timestamp}.png", heatmap_edges, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Vytvorenie kompozitného obrázka so všetkými variantmi (mriežka – 4 stĺpce)
    variants = [
        heatmap_original,
        heatmap_inferno,
        heatmap_plasma,
        heatmap_hot,
        heatmap_blur,
        heatmap_custom
    ]
    composite_all = make_composite(variants, 4)
    cv2.imwrite(f"heatmap_composite_all_{timestamp}.png", composite_all, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print("Export dokončený. Obrázky boli uložené.")

if __name__ == "__main__":
    main()
