import cv2
import numpy as np
import datetime
import matplotlib.cm as mpl_cm
import time

def apply_custom_colormap(gray_img):
    normed = gray_img.astype(np.float32) / 255.0
    colormap = mpl_cm.get_cmap('viridis')
    colored = colormap(normed)
    colored = (colored[:, :, :3] * 255).astype(np.uint8)
    colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    return colored

def make_composite(images, cols):
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

def get_max_resolution(cam_index):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        return (0, 0)
    common_resolutions = [
        (3840, 2160),
        (2560, 1440),
        (1920, 1080),
        (1600, 1200),
        (1280, 720),
        (1024, 768),
        (800, 600),
        (640, 480)
    ]
    for res in common_resolutions:
        w, h = res
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # Dajme pár snímkov na prejavenie nastavenia
        for _ in range(3):
            ret, _ = cap.read()
            time.sleep(0.05)  # krátke oneskorenie
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width >= w and actual_height >= h:
            cap.release()
            return (w, h)
    cap.release()
    return (0, 0)

def list_cameras(max_index=5):  # znížený rozsah, ak vieš, že máš menej kamier
    available = []
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                res = get_max_resolution(i)
                available.append((i, res))
                cap.release()
        except KeyboardInterrupt:
            print("Prerušené prehľadávanie kamier.")
            break
    return available

def main():
    try:
        available_cams = list_cameras(5)
    except KeyboardInterrupt:
        print("Prehľadávanie kamier bolo prerušené.")
        return

    if not available_cams:
        print("Neboli nájdené žiadne pripojené kamery.")
        return

    print("Pripojené kamery:")
    for cam_idx, res in available_cams:
        print(f"  {cam_idx}: Kamera {cam_idx} (max rozlíšenie: {res[0]}x{res[1]})")

    while True:
        try:
            selected = int(input("Zadaj index kamery, ktorú chceš použiť: "))
            if any(cam_idx == selected for cam_idx, _ in available_cams):
                break
            else:
                print("Neplatný index. Skús znova.")
        except ValueError:
            print("Zadaj prosím číslo.")

    cap = cv2.VideoCapture(selected)
    if not cap.isOpened():
        print(f"Chyba: Kamera s indexom {selected} sa nepodarilo otvoriť.")
        return

    desired_width, desired_height = 1920, 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    ret, frame = cap.read()
    if not ret:
        print("Chyba: Nepodarilo sa načítať snímok.")
        return
    height, width = frame.shape[:2]
    print("Používa sa kamera", selected, "s aktuálnym rozlíšením:", width, "x", height)

    accumulator = np.zeros((height, width), dtype=np.float32)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    print("Beží snímanie – stlač ESC pre ukončenie a export heatmapy.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Koniec streamu alebo chyba snímania.")
            break

        fgMask = backSub.apply(frame)
        _, fgMask = cv2.threshold(fgMask, 150, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgMask = cv2.dilate(fgMask, kernel, iterations=1)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        accumulator += fgMask.astype(np.float32) / 255.0

        norm_acc = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_vis = cv2.applyColorMap(norm_acc, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap_vis, 0.4, 0)

        cv2.imshow("Heatmap Overlay", overlay)
        cv2.imshow("Maska pohybu", fgMask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    norm_acc = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_original = cv2.applyColorMap(norm_acc, cv2.COLORMAP_JET)
    heatmap_inferno = cv2.applyColorMap(norm_acc, cv2.COLORMAP_INFERNO)
    heatmap_plasma = cv2.applyColorMap(norm_acc, cv2.COLORMAP_PLASMA)
    heatmap_hot = cv2.applyColorMap(norm_acc, cv2.COLORMAP_HOT)
    heatmap_blur = cv2.GaussianBlur(heatmap_inferno, (11, 11), 0)
    heatmap_custom = apply_custom_colormap(norm_acc)
    edges = cv2.Canny(norm_acc, 50, 150)
    heatmap_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    cv2.imwrite(f"heatmap_original_{timestamp}.png", heatmap_original, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"heatmap_inferno_{timestamp}.png", heatmap_inferno, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"heatmap_plasma_{timestamp}.png", heatmap_plasma, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"heatmap_hot_{timestamp}.png", heatmap_hot, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"heatmap_blur_{timestamp}.png", heatmap_blur, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"heatmap_custom_{timestamp}.png", heatmap_custom, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"heatmap_edges_{timestamp}.png", heatmap_edges, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    variants = [
        heatmap_original,
        heatmap_inferno,
        heatmap_plasma,
        heatmap_hot,
        heatmap_blur,
        heatmap_custom,
        heatmap_edges
    ]
    composite_all = make_composite(variants, 4)
    cv2.imwrite(f"heatmap_composite_all_{timestamp}.png", composite_all, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print("Export dokončený. Obrázky boli uložené.")

if __name__ == "__main__":
    main()
