"""
Практическая работа №7. RGB <-> HSL

Программа для загрузки изображения, ручного преобразования RGB -> HSL,
интерактивной коррекции H/S/L и сохранения результата.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk


SUPPORTED_OPEN_TYPES = (
    ("Изображения", "*.png *.jpg *.jpeg *.bmp"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("BMP", "*.bmp"),
    ("Все файлы", "*.*"),
)

SUPPORTED_SAVE_TYPES = (
    ("PNG", "*.png"),
    ("JPEG", "*.jpg"),
    ("BMP", "*.bmp"),
)


def clamp01(value: np.ndarray) -> np.ndarray:
    """Ограничивает массив значений диапазоном [0; 1]."""
    return np.clip(value, 0.0, 1.0)


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """
    Преобразует RGB в HSL.

    Вход:
        rgb: массив uint8 формы (height, width, 3), значения RGB в диапазоне [0; 255].

    Выход:
        массив float32 формы (height, width, 3):
        H — [0; 360), S — [0; 1], L — [0; 1].
    """
    rgb_norm = rgb.astype(np.float32) / 255.0

    r = rgb_norm[..., 0]
    g = rgb_norm[..., 1]
    b = rgb_norm[..., 2]

    max_c = np.maximum.reduce([r, g, b])
    min_c = np.minimum.reduce([r, g, b])
    delta = max_c - min_c

    h = np.zeros_like(max_c, dtype=np.float32)
    s = np.zeros_like(max_c, dtype=np.float32)
    l = (max_c + min_c) / 2.0

    chromatic = delta > 0.0

    # S = (MAX - MIN) / (MAX + MIN), если 0 < L <= 0.5
    mask_l_low = chromatic & (l <= 0.5)
    s[mask_l_low] = delta[mask_l_low] / (max_c[mask_l_low] + min_c[mask_l_low])

    # S = (MAX - MIN) / (2 - MAX - MIN), если 0.5 < L < 1
    mask_l_high = chromatic & (l > 0.5)
    s[mask_l_high] = delta[mask_l_high] / (2.0 - max_c[mask_l_high] - min_c[mask_l_high])

    # H зависит от канала, в котором находится максимум.
    mask_r = chromatic & (max_c == r)
    mask_g = chromatic & ~mask_r & (max_c == g)
    mask_b = chromatic & ~mask_r & ~mask_g

    # Если MAX = R и G < B, добавляется 360 градусов.
    h_r = 60.0 * ((g - b) / np.where(delta == 0.0, 1.0, delta))
    h[mask_r] = np.where(g[mask_r] >= b[mask_r], h_r[mask_r], h_r[mask_r] + 360.0)

    h[mask_g] = 60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120.0
    h[mask_b] = 60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240.0

    h %= 360.0

    return np.dstack((h, clamp01(s), clamp01(l))).astype(np.float32)


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """
    Преобразует HSL в RGB.

    Вход:
        hsl: массив float формы (height, width, 3):
        H — градусы, S — [0; 1], L — [0; 1].

    Выход:
        массив uint8 формы (height, width, 3), значения RGB в диапазоне [0; 255].
    """
    h = hsl[..., 0] % 360.0
    s = clamp01(hsl[..., 1])
    l = clamp01(hsl[..., 2])

    q = np.where(l < 0.5, l * (1.0 + s), l + s - l * s)
    p = 2.0 * l - q

    h_k = h / 360.0
    t_r = h_k + 1.0 / 3.0
    t_g = h_k
    t_b = h_k - 1.0 / 3.0

    def normalize_t(t: np.ndarray) -> np.ndarray:
        return np.where(t < 0.0, t + 1.0, np.where(t > 1.0, t - 1.0, t))

    def channel_from_t(t: np.ndarray) -> np.ndarray:
        t = normalize_t(t)
        return np.where(
            t < 1.0 / 6.0,
            p + ((q - p) * 6.0 * t),
            np.where(
                t < 1.0 / 2.0,
                q,
                np.where(
                    t < 2.0 / 3.0,
                    p + ((q - p) * (2.0 / 3.0 - t) * 6.0),
                    p,
                ),
            ),
        )

    r = channel_from_t(t_r)
    g = channel_from_t(t_g)
    b = channel_from_t(t_b)

    rgb = np.dstack((r, g, b))
    return np.rint(clamp01(rgb) * 255.0).astype(np.uint8)


def hsl_to_rgb_color(hue: float, saturation: float = 1.0, lightness: float = 0.5) -> tuple[int, int, int]:
    """
    Возвращает один RGB-цвет для индикатора текущего тона H.

    Вход:
        hue: тон в градусах.
        saturation: насыщенность [0; 1].
        lightness: светлота [0; 1].

    Выход:
        кортеж (R, G, B), значения [0; 255].
    """
    hsl_pixel = np.array([[[hue, saturation, lightness]]], dtype=np.float32)
    rgb = hsl_to_rgb(hsl_pixel)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


class HSLImageEditor(tk.Tk):
    """Главное окно программы."""

    def __init__(self) -> None:
        super().__init__()

        self.title("RGB ↔ HSL")
        self.minsize(960, 650)

        self.original_image: Image.Image | None = None
        self.original_hsl: np.ndarray | None = None
        self.alpha_channel: np.ndarray | None = None
        self.result_image: Image.Image | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.update_job: str | None = None

        self._build_interface()

    def _build_interface(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top_panel = ttk.Frame(self, padding=10)
        top_panel.grid(row=0, column=0, sticky="ew")
        top_panel.columnconfigure(3, weight=1)

        ttk.Button(top_panel, text="Открыть изображение", command=self.open_image).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(top_panel, text="Сбросить ползунки", command=self.reset_sliders).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(top_panel, text="Сохранить результат", command=self.save_image).grid(row=0, column=2, padx=(0, 8))

        self.file_label = ttk.Label(top_panel, text="Файл не выбран")
        self.file_label.grid(row=0, column=3, sticky="w")

        image_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        image_frame.grid(row=1, column=0, sticky="nsew")
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(
            image_frame,
            text="Загрузите PNG, JPEG или BMP изображение",
            anchor="center",
            relief="groove",
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")

        controls = ttk.LabelFrame(self, text="Параметры HSL", padding=10)
        controls.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        controls.columnconfigure(1, weight=1)

        self.hue_value = tk.IntVar(value=0)
        self.saturation_value = tk.IntVar(value=100)
        self.lightness_value = tk.IntVar(value=0)

        ttk.Label(controls, text="Hue, сдвиг тона:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.hue_scale = ttk.Scale(
            controls,
            from_=0,
            to=360,
            variable=self.hue_value,
            command=lambda _value: self.schedule_preview_update(),
        )
        self.hue_scale.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        self.hue_label = ttk.Label(controls, width=10)
        self.hue_label.grid(row=0, column=2, sticky="e")

        self.hue_indicator = tk.Canvas(controls, width=70, height=28, highlightthickness=1, highlightbackground="#777777")
        self.hue_indicator.grid(row=0, column=3, padx=(12, 0))
        self.hue_indicator_rect = self.hue_indicator.create_rectangle(0, 0, 70, 28, outline="", fill="#ff0000")

        ttk.Label(controls, text="Saturation, множитель:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.saturation_scale = ttk.Scale(
            controls,
            from_=0,
            to=200,
            variable=self.saturation_value,
            command=lambda _value: self.schedule_preview_update(),
        )
        self.saturation_scale.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(8, 0))

        self.saturation_label = ttk.Label(controls, width=10)
        self.saturation_label.grid(row=1, column=2, sticky="e", pady=(8, 0))

        ttk.Label(controls, text="Lightness, добавка:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.lightness_scale = ttk.Scale(
            controls,
            from_=-100,
            to=100,
            variable=self.lightness_value,
            command=lambda _value: self.schedule_preview_update(),
        )
        self.lightness_scale.grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=(8, 0))

        self.lightness_label = ttk.Label(controls, width=10)
        self.lightness_label.grid(row=2, column=2, sticky="e", pady=(8, 0))

        hint = (
            "H изменяет тон по кругу, S усиливает/ослабляет насыщенность, "
            "L делает изображение темнее или светлее. Индикатор справа показывает текущий Hue при S=100% и L=50%."
        )
        ttk.Label(controls, text=hint, foreground="#555555").grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))

        self.bind("<Configure>", lambda _event: self.schedule_preview_update(only_resize=True))
        self.update_labels_and_indicator()

    def open_image(self) -> None:
        path = filedialog.askopenfilename(title="Выберите изображение", filetypes=SUPPORTED_OPEN_TYPES)
        if not path:
            return

        try:
            image = Image.open(path).convert("RGBA")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть изображение:\n{exc}")
            return

        rgba = np.array(image, dtype=np.uint8)
        self.original_image = image
        self.original_hsl = rgb_to_hsl(rgba[..., :3])
        self.alpha_channel = rgba[..., 3]
        self.file_label.config(text=os.path.basename(path))
        self.reset_sliders()
        self.update_preview()

    def reset_sliders(self) -> None:
        self.hue_value.set(0)
        self.saturation_value.set(100)
        self.lightness_value.set(0)
        self.update_preview()

    def schedule_preview_update(self, only_resize: bool = False) -> None:
        if self.update_job is not None:
            self.after_cancel(self.update_job)

        delay_ms = 80 if only_resize else 25
        self.update_job = self.after(delay_ms, self.update_preview)

    def update_labels_and_indicator(self) -> None:
        hue = int(round(self.hue_value.get())) % 360
        sat = int(round(self.saturation_value.get()))
        light = int(round(self.lightness_value.get()))

        self.hue_label.config(text=f"{hue}°")
        self.saturation_label.config(text=f"{sat}%")
        self.lightness_label.config(text=f"{light:+d}%")

        r, g, b = hsl_to_rgb_color(hue, 1.0, 0.5)
        color_hex = f"#{r:02x}{g:02x}{b:02x}"
        self.hue_indicator.itemconfig(self.hue_indicator_rect, fill=color_hex)

    def make_result_image(self) -> Image.Image | None:
        if self.original_hsl is None or self.alpha_channel is None:
            return None

        hue_shift = float(self.hue_value.get())
        saturation_factor = float(self.saturation_value.get()) / 100.0
        lightness_delta = float(self.lightness_value.get()) / 100.0

        edited_hsl = self.original_hsl.copy()
        edited_hsl[..., 0] = (edited_hsl[..., 0] + hue_shift) % 360.0
        edited_hsl[..., 1] = clamp01(edited_hsl[..., 1] * saturation_factor)
        edited_hsl[..., 2] = clamp01(edited_hsl[..., 2] + lightness_delta)

        rgb = hsl_to_rgb(edited_hsl)
        rgba = np.dstack((rgb, self.alpha_channel)).astype(np.uint8)
        return Image.fromarray(rgba, mode="RGBA")

    def update_preview(self) -> None:
        self.update_job = None
        self.update_labels_and_indicator()

        image = self.make_result_image()
        if image is None:
            return

        self.result_image = image

        label_width = max(self.image_label.winfo_width(), 400)
        label_height = max(self.image_label.winfo_height(), 300)
        preview = image.copy()
        preview.thumbnail((label_width - 20, label_height - 20), Image.Resampling.LANCZOS)

        self.preview_photo = ImageTk.PhotoImage(preview)
        self.image_label.config(image=self.preview_photo, text="")

    def save_image(self) -> None:
        if self.result_image is None:
            messagebox.showwarning("Нет изображения", "Сначала загрузите изображение.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить изображение",
            defaultextension=".png",
            filetypes=SUPPORTED_SAVE_TYPES,
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()

        try:
            if ext in {".jpg", ".jpeg"}:
                # JPEG не поддерживает прозрачность, поэтому альфа-канал заменяется белым фоном.
                background = Image.new("RGB", self.result_image.size, (255, 255, 255))
                background.paste(self.result_image, mask=self.result_image.getchannel("A"))
                background.save(path, quality=95)
            else:
                self.result_image.save(path)

            messagebox.showinfo("Готово", f"Изображение сохранено:\n{path}")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось сохранить изображение:\n{exc}")


def main() -> None:
    app = HSLImageEditor()
    app.mainloop()


if __name__ == "__main__":
    main()
