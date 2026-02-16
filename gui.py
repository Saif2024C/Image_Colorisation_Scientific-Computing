import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from generalfunctions import get_D, random_bool_array, uniform_bool_array
from greyscaleconversion import RGB_to_greyscale, generate_mixed_img
from recolorizer import recolorise

# -------------------------
# Adjustable GUI parameters
# -------------------------
APP_TITLE = "Image Recolorizer"
WINDOW_BG = "#e8ecef"
IMAGE_FRAME_BG = "#d8dee4"
BUTTON_COLUMN_BG = "#e8ecef"
SUBIMAGE_BG = "#ffffff"
SUBIMAGE_BORDER_COLOR = "#9aa5b1"
BUTTON_BG = "#2f6f94"
BUTTON_FG = "#111111"
BUTTON_ACTIVE_BG = "#255a78"
STATUS_FG = "#1f2933"
PANEL_BORDER_WIDTH = 1

OUTER_PAD_X = 18
OUTER_PAD_Y = 18
IMAGE_FRAME_PAD_X = 12
IMAGE_FRAME_PAD_Y = 12
BUTTON_COLUMN_PAD_X = 12
BUTTON_COLUMN_PAD_Y = 12
GRID_GAP_X = 10
GRID_GAP_Y = 10
BUTTON_GAP_Y = 10
BUTTON_WIDTH = 22

CELL_WIDTH = 320
CELL_HEIGHT = 240
RANDOM_COLORPOINT_COUNT = 500
DEFAULT_W_INTERVAL = 10
DEFAULT_H_INTERVAL = 10
FOREGROUND_DELAY_MS = 50
FRAME_MARGIN_RATIO = 0.02
PANEL_GAP_RATIO = 0.01
MIN_FRAME_MARGIN = 8
MIN_PANEL_GAP = 4
DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 800
GEOMETRY_ENFORCE_DELAY_MS = 120
DEFAULT_GREYSCALE_METHOD = "Luminosity"
DEFAULT_MANUAL_BRUSH_SIZE = 1.0
MIN_MANUAL_BRUSH_SIZE = 0.1
MAX_MANUAL_BRUSH_SIZE = 10.0

try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS


class RecolorizerApp(tk.Tk):
    def __init__(self) -> None:
        """Initialise the GUI window, state variables, and base widgets.

        Returns:
            None.
        """
        super().__init__()

        self.title(APP_TITLE)
        self.configure(bg=WINDOW_BG)

        self.image_source: Image.Image | None = None
        self.image_greyscale: Image.Image | None = None
        self.image_mixed: Image.Image | None = None
        self.image_recolorised: Image.Image | None = None
        self.colorpoints_coords = None
        self.colorpoint_mode_used = None
        self.greyscale_method_used = None
        self.manual_colorpoints_coords = None
        self.manual_color_array = None
        self.manual_greyscale_array = None
        self.manual_mixed_array = None
        self.manual_method_used = None
        self.manual_drag_active = False
        self.manual_last_point = None
        self.manual_sparse_counter = 0
        self.panel_aspect_ratio = CELL_WIDTH / CELL_HEIGHT
        self.frame_margin_x = IMAGE_FRAME_PAD_X
        self.frame_margin_y = IMAGE_FRAME_PAD_Y
        self.panel_gap_x = GRID_GAP_X
        self.panel_gap_y = GRID_GAP_Y

        self._photo_refs = {}
        self._panel_images = {}

        self._build_layout()
        self._set_all_placeholders()
        self.after_idle(self._set_default_window_size)
        self.after(GEOMETRY_ENFORCE_DELAY_MS, self._set_default_window_size)
        self.after(FOREGROUND_DELAY_MS, self._bring_to_foreground)

    def _build_layout(self) -> None:
        """Create the main GUI layout and place all widgets.

        Returns:
            None.
        """
        root = tk.Frame(self, bg=WINDOW_BG)
        root.pack(fill="both", expand=True, padx=OUTER_PAD_X, pady=OUTER_PAD_Y)

        image_frame = tk.Frame(root, bg=IMAGE_FRAME_BG, bd=0)
        image_frame.pack(side="left", fill="both", expand=True)

        self.image_grid = tk.Frame(
            image_frame,
            bg=IMAGE_FRAME_BG,
            padx=IMAGE_FRAME_PAD_X,
            pady=IMAGE_FRAME_PAD_Y,
            highlightthickness=1,
            highlightbackground=SUBIMAGE_BORDER_COLOR,
        )
        self.image_grid.pack(fill="both", expand=True)

        for row in range(2):
            self.image_grid.grid_rowconfigure(
                row,
                weight=1,
                minsize=CELL_HEIGHT + (2 * GRID_GAP_Y),
            )
        for col in range(2):
            self.image_grid.grid_columnconfigure(
                col,
                weight=1,
                minsize=CELL_WIDTH + (2 * GRID_GAP_X),
            )

        self.source_canvas = self._create_image_panel(self.image_grid, 0, 0)
        self.greyscale_canvas = self._create_image_panel(self.image_grid, 0, 1)
        self.mixed_canvas = self._create_image_panel(self.image_grid, 1, 0)
        self.recolorised_canvas = self._create_image_panel(self.image_grid, 1, 1)

        self.panels = {
            "source": self.source_canvas,
            "greyscale": self.greyscale_canvas,
            "mixed": self.mixed_canvas,
            "recolorised": self.recolorised_canvas,
        }
        self.image_grid.bind("<Configure>", self._on_image_grid_resize)
        for key, panel in self.panels.items():
            panel.bind("<Configure>", lambda _event, name=key: self._on_panel_resize(name))
        self.source_canvas.bind("<ButtonPress-1>", self._on_source_mouse_down)
        self.source_canvas.bind("<B1-Motion>", self._on_source_mouse_drag)
        self.source_canvas.bind("<ButtonRelease-1>", self._on_source_mouse_up)
        self.after_idle(self._update_panel_sizes)

        button_column = tk.Frame(
            root,
            bg=BUTTON_COLUMN_BG,
            padx=BUTTON_COLUMN_PAD_X,
            pady=BUTTON_COLUMN_PAD_Y,
        )
        button_column.pack(side="right", fill="y")

        self._create_button(button_column, "Load Source Image", self.load_source_image)
        self._build_greyscale_controls(button_column)
        self._create_button(button_column, "Generate Greyscale", self.create_greyscale_image)
        self._build_colorpoint_controls(button_column)
        self.mixed_button = self._create_button(
            button_column,
            "Generate Mixed Image",
            self.create_mixed_image,
        )
        self.recolorised_button = self._create_button(
            button_column,
            "Generate Recolorised",
            self.create_recolorised_image,
        )

        self.status_var = tk.StringVar(value="Load an image to begin.")
        status_label = tk.Label(
            button_column,
            textvariable=self.status_var,
            bg=BUTTON_COLUMN_BG,
            fg=STATUS_FG,
            wraplength=190,
            justify="left",
            anchor="nw",
        )
        status_label.pack(fill="x", pady=(BUTTON_GAP_Y + 2, 0))
        self._create_button(
            button_column,
            "Clear Images",
            self.clear_images,
            side="bottom",
        )
        self._set_mixed_button_visibility()

    def _create_image_panel(self, parent: tk.Widget, row: int, col: int) -> tk.Canvas:
        """Create one fixed-size subimage panel in the 2x2 display grid.

        Args:
            parent: Parent widget containing the image grid.
            row: Grid row index.
            col: Grid column index.

        Returns:
            The created Tkinter canvas.
        """
        panel = tk.Canvas(
            parent,
            bg=SUBIMAGE_BG,
            bd=0,
            width=CELL_WIDTH,
            height=CELL_HEIGHT,
            highlightthickness=PANEL_BORDER_WIDTH,
            highlightbackground=SUBIMAGE_BORDER_COLOR,
        )
        panel.grid(row=row, column=col, padx=GRID_GAP_X, pady=GRID_GAP_Y)
        return panel

    def _create_button(self, parent: tk.Widget, text: str, command, side: str = "top") -> tk.Button:
        """Create one control button in the right-hand column.

        Args:
            parent: Parent widget for the button.
            text: Text shown on the button.
            command: Callback function triggered on button press.
            side: Pack side for placing the button.

        Returns:
            The created button widget.
        """
        button = tk.Button(
            parent,
            text=text,
            command=command,
            width=BUTTON_WIDTH,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_FG,
            relief="flat",
            padx=8,
            pady=8,
        )
        if side == "bottom":
            button.pack(side="bottom", fill="x", pady=(BUTTON_GAP_Y, 0))
        else:
            button.pack(fill="x", pady=(0, BUTTON_GAP_Y))
        return button

    def _build_greyscale_controls(self, parent: tk.Widget) -> None:
        """Build the dropdown for selecting a greyscale conversion method.

        Args:
            parent: Parent widget for the controls.

        Returns:
            None.
        """
        controls_frame = tk.Frame(parent, bg=BUTTON_COLUMN_BG)
        controls_frame.pack(fill="x", pady=(0, BUTTON_GAP_Y))

        method_label = tk.Label(
            controls_frame,
            text="Greyscale method",
            bg=BUTTON_COLUMN_BG,
            fg=STATUS_FG,
            anchor="w",
            justify="left",
        )
        method_label.pack(fill="x", pady=(0, 4))

        self.greyscale_method_var = tk.StringVar(value=DEFAULT_GREYSCALE_METHOD)
        method_dropdown = ttk.Combobox(
            controls_frame,
            textvariable=self.greyscale_method_var,
            values=("Luminosity", "Average", "Luminance", "Lightness"),
            state="readonly",
            width=BUTTON_WIDTH,
        )
        method_dropdown.pack(fill="x")
        self.greyscale_method_var.trace_add("write", self._on_greyscale_method_change)

    def _get_selected_greyscale_method_code(self) -> str:
        """Get the method code corresponding to the selected greyscale dropdown option.

        Returns:
            One of the greyscale method codes: "nor", "avg", "lum", or "lig".
        """
        method_map = {
            "Luminosity": "nor",
            "Average": "avg",
            "Luminance": "lum",
            "Lightness": "lig",
        }
        selected_method = self.greyscale_method_var.get()
        return method_map.get(selected_method, "nor")

    def _build_colorpoint_controls(self, parent: tk.Widget) -> None:
        """Build the mode dropdown and input fields for colorpoint selection.

        Args:
            parent: Parent widget for the controls.

        Returns:
            None.
        """
        controls_frame = tk.Frame(parent, bg=BUTTON_COLUMN_BG)
        controls_frame.pack(fill="x", pady=(0, BUTTON_GAP_Y))

        mode_label = tk.Label(
            controls_frame,
            text="Point selection method",
            bg=BUTTON_COLUMN_BG,
            fg=STATUS_FG,
            anchor="w",
            justify="left",
        )
        mode_label.pack(fill="x", pady=(0, 4))

        self.colorpoint_mode_var = tk.StringVar(value="Random")
        self.random_points_var = tk.StringVar(value=str(RANDOM_COLORPOINT_COUNT))
        self.uniform_w_interval_var = tk.StringVar(value=str(DEFAULT_W_INTERVAL))
        self.uniform_h_interval_var = tk.StringVar(value=str(DEFAULT_H_INTERVAL))

        mode_dropdown = ttk.Combobox(
            controls_frame,
            textvariable=self.colorpoint_mode_var,
            values=("Random", "Uniform", "Manual"),
            state="readonly",
            width=BUTTON_WIDTH,
        )
        mode_dropdown.pack(fill="x", pady=(0, 6))

        self.random_input_frame = tk.Frame(controls_frame, bg=BUTTON_COLUMN_BG)
        random_label = tk.Label(
            self.random_input_frame,
            text="Number of colorpoints",
            bg=BUTTON_COLUMN_BG,
            fg=STATUS_FG,
            anchor="w",
            justify="left",
        )
        random_label.pack(fill="x", pady=(0, 2))
        random_entry = tk.Entry(self.random_input_frame, textvariable=self.random_points_var)
        random_entry.pack(fill="x")

        self.uniform_input_frame = tk.Frame(controls_frame, bg=BUTTON_COLUMN_BG)
        w_interval_label = tk.Label(
            self.uniform_input_frame,
            text="Width interval (pixels)",
            bg=BUTTON_COLUMN_BG,
            fg=STATUS_FG,
            anchor="w",
            justify="left",
        )
        w_interval_label.pack(fill="x", pady=(0, 2))
        w_interval_entry = tk.Entry(self.uniform_input_frame, textvariable=self.uniform_w_interval_var)
        w_interval_entry.pack(fill="x", pady=(0, 6))

        h_interval_label = tk.Label(
            self.uniform_input_frame,
            text="Height interval (pixels)",
            bg=BUTTON_COLUMN_BG,
            fg=STATUS_FG,
            anchor="w",
            justify="left",
        )
        h_interval_label.pack(fill="x", pady=(0, 2))
        h_interval_entry = tk.Entry(self.uniform_input_frame, textvariable=self.uniform_h_interval_var)
        h_interval_entry.pack(fill="x")
        self.uniform_pointcount_var = tk.StringVar(value="Number of colorpoints: load a source image.")
        uniform_pointcount_label = tk.Label(
            self.uniform_input_frame,
            textvariable=self.uniform_pointcount_var,
            bg=BUTTON_COLUMN_BG,
            fg=STATUS_FG,
            anchor="w",
            justify="left",
            wraplength=220,
        )
        uniform_pointcount_label.pack(fill="x", pady=(6, 0))

        self.manual_input_frame = tk.Frame(controls_frame, bg=BUTTON_COLUMN_BG)
        manual_label = tk.Label(
            self.manual_input_frame,
            text="Click and drag on the source image to select colorpoints.",
            bg=BUTTON_COLUMN_BG,
            fg=STATUS_FG,
            anchor="w",
            justify="left",
            wraplength=220,
        )
        manual_label.pack(fill="x", pady=(0, 6))
        self.manual_brush_size_var = tk.DoubleVar(value=DEFAULT_MANUAL_BRUSH_SIZE)
        brush_size_label = tk.Label(
            self.manual_input_frame,
            text="Brush size",
            bg=BUTTON_COLUMN_BG,
            fg=STATUS_FG,
            anchor="w",
            justify="left",
        )
        brush_size_label.pack(fill="x", pady=(0, 2))
        brush_size_scale = tk.Scale(
            self.manual_input_frame,
            variable=self.manual_brush_size_var,
            from_=MIN_MANUAL_BRUSH_SIZE,
            to=MAX_MANUAL_BRUSH_SIZE,
            resolution=0.1,
            orient="horizontal",
            bg=BUTTON_COLUMN_BG,
            fg=STATUS_FG,
            highlightthickness=0,
            relief="flat",
        )
        brush_size_scale.pack(fill="x", pady=(0, 6))
        self.manual_brush_size_text_var = tk.StringVar(value="")
        brush_size_value_label = tk.Label(
            self.manual_input_frame,
            textvariable=self.manual_brush_size_text_var,
            bg=BUTTON_COLUMN_BG,
            fg=STATUS_FG,
            anchor="w",
            justify="left",
        )
        brush_size_value_label.pack(fill="x", pady=(0, 6))
        manual_reset_button = tk.Button(
            self.manual_input_frame,
            text="Reset Manual Points",
            command=self._reset_manual_points,
            width=BUTTON_WIDTH,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_FG,
            relief="flat",
            padx=8,
            pady=6,
        )
        manual_reset_button.pack(fill="x")
        self.manual_pointcount_var = tk.StringVar(value="Manual colorpoints selected: 0")
        manual_pointcount_label = tk.Label(
            self.manual_input_frame,
            textvariable=self.manual_pointcount_var,
            bg=BUTTON_COLUMN_BG,
            fg=STATUS_FG,
            anchor="w",
            justify="left",
            wraplength=220,
        )
        manual_pointcount_label.pack(fill="x", pady=(6, 0))

        self.colorpoint_mode_var.trace_add("write", self._on_colorpoint_mode_change)
        self.uniform_w_interval_var.trace_add("write", self._on_uniform_interval_change)
        self.uniform_h_interval_var.trace_add("write", self._on_uniform_interval_change)
        self.manual_brush_size_var.trace_add("write", self._on_manual_brush_size_change)
        self._update_uniform_pointcount_text()
        self._update_manual_brush_size_text()
        self._update_manual_pointcount_text()
        self._update_colorpoint_input_fields()

    def _on_colorpoint_mode_change(self, *_args) -> None:
        """Handle updates when the colorpoint mode dropdown value changes.

        Args:
            *_args: Tkinter trace arguments.

        Returns:
            None.
        """
        self._update_colorpoint_input_fields()
        self._update_manual_pointcount_text()
        self._set_mixed_button_visibility()
        if self.colorpoint_mode_var.get() == "Manual":
            if self.image_source is None:
                self.status_var.set("Load a source image, then click and drag to select manual points.")
                return
            if self._ensure_manual_selection_ready():
                self.status_var.set("Manual mode active. Click and drag on source image to select points.")

    def _set_mixed_button_visibility(self) -> None:
        """Show or hide the mixed-image button based on the selected point mode.

        Returns:
            None.
        """
        is_manual_mode = self.colorpoint_mode_var.get() == "Manual"
        is_packed = self.mixed_button.winfo_manager() == "pack"

        if is_manual_mode and is_packed:
            self.mixed_button.pack_forget()
            return

        if not is_manual_mode and not is_packed:
            self.mixed_button.pack(fill="x", pady=(0, BUTTON_GAP_Y), before=self.recolorised_button)

    def _on_greyscale_method_change(self, *_args) -> None:
        """Handle updates when the greyscale method dropdown value changes.

        Args:
            *_args: Tkinter trace arguments.

        Returns:
            None.
        """
        if self.colorpoint_mode_var.get() != "Manual":
            return

        if self.image_source is None or self.manual_colorpoints_coords is None:
            return

        self._rebuild_manual_mixed_from_mask()

    def _on_uniform_interval_change(self, *_args) -> None:
        """Handle updates when the uniform interval inputs are edited.

        Args:
            *_args: Tkinter trace arguments.

        Returns:
            None.
        """
        self._update_uniform_pointcount_text()

    def _on_manual_brush_size_change(self, *_args) -> None:
        """Handle updates when the manual brush-size control value changes.

        Args:
            *_args: Tkinter trace arguments.

        Returns:
            None.
        """
        self._update_manual_brush_size_text()

    def _update_colorpoint_input_fields(self) -> None:
        """Show only the input fields relevant to the selected colorpoint mode.

        Returns:
            None.
        """
        mode = self.colorpoint_mode_var.get()
        self.random_input_frame.pack_forget()
        self.uniform_input_frame.pack_forget()
        self.manual_input_frame.pack_forget()

        if mode == "Uniform":
            self.uniform_input_frame.pack(fill="x")
        elif mode == "Manual":
            self.manual_input_frame.pack(fill="x")
        else:
            self.random_input_frame.pack(fill="x")

    def _count_uniform_colorpoints(
        self,
        width: int,
        height: int,
        w_interval: int,
        h_interval: int,
    ) -> int:
        """Count uniform colorpoints implied by image size and interval values.

        Args:
            width: Source image width.
            height: Source image height.
            w_interval: Width interval used in uniform sampling.
            h_interval: Height interval used in uniform sampling.

        Returns:
            Number of selected colorpoints.
        """
        points_width = ((width - 1) // (w_interval + 1)) + 1
        points_height = ((height - 1) // (h_interval + 1)) + 1
        return points_width * points_height

    def _update_uniform_pointcount_text(self) -> None:
        """Update the displayed number of colorpoints for uniform interval inputs.

        Returns:
            None.
        """
        if self.image_source is None:
            self.uniform_pointcount_var.set("Number of colorpoints: load a source image.")
            return

        try:
            w_interval = int(self.uniform_w_interval_var.get())
            h_interval = int(self.uniform_h_interval_var.get())
        except ValueError:
            self.uniform_pointcount_var.set("Number of colorpoints: enter integer intervals.")
            return

        if w_interval < 0 or h_interval < 0:
            self.uniform_pointcount_var.set("Number of colorpoints: intervals must be non-negative.")
            return

        width, height = self.image_source.size
        number_of_points = self._count_uniform_colorpoints(width, height, w_interval, h_interval)
        self.uniform_pointcount_var.set(f"Number of colorpoints: {number_of_points}")

    def _is_manual_mode(self) -> bool:
        """Check whether the manual point-selection mode is currently selected.

        Returns:
            True if manual mode is selected, otherwise False.
        """
        return self.colorpoint_mode_var.get() == "Manual"

    def _get_manual_brush_size(self) -> float:
        """Get a validated manual brush-size value.

        Returns:
            Brush size in the range [MIN_MANUAL_BRUSH_SIZE, MAX_MANUAL_BRUSH_SIZE].
        """
        brush_size = float(self.manual_brush_size_var.get())
        return min(MAX_MANUAL_BRUSH_SIZE, max(MIN_MANUAL_BRUSH_SIZE, brush_size))

    def _update_manual_brush_size_text(self) -> None:
        """Update the manual brush-size helper text.

        Returns:
            None.
        """
        if not hasattr(self, "manual_brush_size_text_var"):
            return

        brush_size = self._get_manual_brush_size()
        if brush_size < 1:
            stride = max(1, int(round(1.0 / brush_size)))
            self.manual_brush_size_text_var.set(
                f"Brush size: {brush_size:.1f}"
            )
        else:
            radius = int(round(brush_size))
            self.manual_brush_size_text_var.set(
                f"Brush size: {brush_size:.1f}"
            )

    def _get_manual_selected_point_count(self) -> int:
        """Get the current number of manually selected colorpoints.

        Returns:
            Number of selected manual points.
        """
        if self.manual_colorpoints_coords is None:
            return 0

        return int(np.count_nonzero(self.manual_colorpoints_coords[:, :, 0]))

    def _update_manual_pointcount_text(self) -> None:
        """Update the displayed number of manually selected colorpoints.

        Returns:
            None.
        """
        if not hasattr(self, "manual_pointcount_var"):
            return

        if self.image_source is None:
            self.manual_pointcount_var.set("Manual colorpoints: load a source image.")
            return

        number_of_points = self._get_manual_selected_point_count()
        self.manual_pointcount_var.set(f"Manual colorpoints selected: {number_of_points}")

    def _reset_manual_state(self) -> None:
        """Reset all cached data related to manual point selection.

        Returns:
            None.
        """
        self.manual_colorpoints_coords = None
        self.manual_color_array = None
        self.manual_greyscale_array = None
        self.manual_mixed_array = None
        self.manual_method_used = None
        self.manual_drag_active = False
        self.manual_last_point = None
        self.manual_sparse_counter = 0
        self._update_manual_pointcount_text()

    def _reset_manual_points(self) -> None:
        """Clear currently selected manual points and refresh the mixed-image preview.

        Returns:
            None.
        """
        if self.image_source is None:
            self.status_var.set("Load a source image before resetting manual points.")
            return

        if not self._is_manual_mode():
            return

        self._ensure_manual_selection_ready(reset=True)
        self.colorpoints_coords = None
        self.colorpoint_mode_used = None
        self._update_manual_pointcount_text()
        self.status_var.set("Manual points reset.")

    def _ensure_manual_selection_ready(self, reset: bool = False) -> bool:
        """Ensure manual-selection arrays are initialised for the current source image.

        Args:
            reset: Whether to force-reset all manual selections.

        Returns:
            True if manual selection data is ready, otherwise False.
        """
        if self.image_source is None:
            return False

        width, height = self.image_source.size
        method_code = self._get_selected_greyscale_method_code()

        arrays_missing = (
            self.manual_colorpoints_coords is None
            or self.manual_color_array is None
            or self.manual_greyscale_array is None
            or self.manual_mixed_array is None
        )

        shape_changed = (
            self.manual_colorpoints_coords is not None
            and self.manual_colorpoints_coords.shape != (height, width, 3)
        )

        method_changed = self.manual_method_used != method_code

        if reset or arrays_missing or shape_changed:
            self.manual_colorpoints_coords = np.zeros((height, width, 3), dtype=bool)
            self.manual_color_array = np.asarray(self.image_source).copy()
            greyscale_image = RGB_to_greyscale(self.image_source, method_code)
            self.manual_greyscale_array = np.asarray(greyscale_image).copy()
            self.manual_mixed_array = self.manual_greyscale_array.copy()
            self.manual_method_used = method_code
            self.manual_sparse_counter = 0
            self.colorpoints_coords = None
            self.colorpoint_mode_used = None
            self.image_mixed = Image.fromarray(self.manual_mixed_array.copy())
            self._set_subimage("mixed", self.image_mixed)
            self._update_manual_pointcount_text()
            return True

        if method_changed:
            self._rebuild_manual_mixed_from_mask()

        return True

    def _rebuild_manual_mixed_from_mask(self) -> None:
        """Rebuild the manual mixed-image preview from the current point mask and method.

        Returns:
            None.
        """
        if self.image_source is None or self.manual_colorpoints_coords is None:
            return

        method_code = self._get_selected_greyscale_method_code()
        self.manual_color_array = np.asarray(self.image_source).copy()
        greyscale_image = RGB_to_greyscale(self.image_source, method_code)
        self.manual_greyscale_array = np.asarray(greyscale_image).copy()
        self.manual_mixed_array = np.where(
            self.manual_colorpoints_coords,
            self.manual_color_array,
            self.manual_greyscale_array,
        )
        self.manual_method_used = method_code
        self.image_mixed = Image.fromarray(self.manual_mixed_array.copy())
        self._set_subimage("mixed", self.image_mixed)
        self._update_manual_pointcount_text()

    def _panel_to_source_point(self, panel_x: int, panel_y: int):
        """Map a source-panel mouse position to a source-image pixel coordinate.

        Args:
            panel_x: X-coordinate in source-panel pixels.
            panel_y: Y-coordinate in source-panel pixels.

        Returns:
            A tuple (x, y) in source-image pixels, or None if out of bounds.
        """
        if self.image_source is None:
            return None

        panel_width = max(self.source_canvas.winfo_width(), 1)
        panel_height = max(self.source_canvas.winfo_height(), 1)
        if panel_x < 0 or panel_y < 0 or panel_x >= panel_width or panel_y >= panel_height:
            return None

        image_width, image_height = self.image_source.size
        source_x = min(image_width - 1, int((panel_x / panel_width) * image_width))
        source_y = min(image_height - 1, int((panel_y / panel_height) * image_height))
        return source_x, source_y

    def _apply_manual_brush(self, source_x: int, source_y: int) -> None:
        """Apply the manual-selection brush around a source-image point.

        Args:
            source_x: X-coordinate in source-image pixels.
            source_y: Y-coordinate in source-image pixels.

        Returns:
            None.
        """
        if self.manual_colorpoints_coords is None or self.manual_mixed_array is None:
            return

        brush_size = self._get_manual_brush_size()
        image_height, image_width = self.manual_colorpoints_coords.shape[:2]
        if brush_size < 1:
            stride = max(1, int(round(1.0 / brush_size)))
            if self.manual_sparse_counter % stride != 0:
                self.manual_sparse_counter += 1
                return

            self.manual_sparse_counter += 1
            self.manual_colorpoints_coords[source_y:source_y + 1, source_x:source_x + 1, :] = True
            self.manual_mixed_array[source_y:source_y + 1, source_x:source_x + 1, :] = (
                self.manual_color_array[source_y:source_y + 1, source_x:source_x + 1, :]
            )
            return

        brush_radius = int(round(brush_size))
        x_start = max(0, source_x - brush_radius)
        x_end = min(image_width, source_x + brush_radius + 1)
        y_start = max(0, source_y - brush_radius)
        y_end = min(image_height, source_y + brush_radius + 1)

        self.manual_colorpoints_coords[y_start:y_end, x_start:x_end, :] = True
        self.manual_mixed_array[y_start:y_end, x_start:x_end, :] = self.manual_color_array[
            y_start:y_end,
            x_start:x_end,
            :,
        ]

    def _apply_manual_line(self, start_point, end_point) -> None:
        """Apply manual selection along a line between two source-image points.

        Args:
            start_point: Start point tuple (x, y).
            end_point: End point tuple (x, y).

        Returns:
            None.
        """
        x0, y0 = start_point
        x1, y1 = end_point
        steps = max(abs(x1 - x0), abs(y1 - y0)) + 1

        if steps <= 1:
            self._apply_manual_brush(x1, y1)
            return

        for step in range(steps):
            t = step / (steps - 1)
            x = int(round(x0 + (x1 - x0) * t))
            y = int(round(y0 + (y1 - y0) * t))
            self._apply_manual_brush(x, y)

    def _on_source_mouse_down(self, event) -> None:
        """Handle the start of a manual point-selection drag on the source panel.

        Args:
            event: Tkinter mouse event.

        Returns:
            None.
        """
        if not self._is_manual_mode():
            return

        if not self._ensure_manual_selection_ready():
            return

        point = self._panel_to_source_point(event.x, event.y)
        if point is None:
            return

        self.manual_drag_active = True
        self.manual_sparse_counter = 0
        self.manual_last_point = point
        self._apply_manual_brush(point[0], point[1])
        self.colorpoints_coords = self.manual_colorpoints_coords.copy()
        self.colorpoint_mode_used = "Manual"
        self.image_mixed = Image.fromarray(self.manual_mixed_array.copy())
        self._set_subimage("mixed", self.image_mixed)
        self._update_manual_pointcount_text()

    def _on_source_mouse_drag(self, event) -> None:
        """Handle mouse dragging for manual point selection on the source panel.

        Args:
            event: Tkinter mouse event.

        Returns:
            None.
        """
        if not self._is_manual_mode():
            return

        if not self.manual_drag_active:
            return

        point = self._panel_to_source_point(event.x, event.y)
        if point is None:
            return

        if self.manual_last_point is None:
            self.manual_last_point = point

        self._apply_manual_line(self.manual_last_point, point)
        self.manual_last_point = point
        self.colorpoints_coords = self.manual_colorpoints_coords.copy()
        self.colorpoint_mode_used = "Manual"
        self.image_mixed = Image.fromarray(self.manual_mixed_array.copy())
        self._set_subimage("mixed", self.image_mixed)
        self._update_manual_pointcount_text()

    def _on_source_mouse_up(self, _event) -> None:
        """Handle the end of a manual point-selection drag on the source panel.

        Args:
            _event: Tkinter mouse event.

        Returns:
            None.
        """
        self.manual_drag_active = False
        self.manual_last_point = None

    def _read_int(self, value: str, field_name: str, minimum: int):
        """Parse and validate an integer value from a text field.

        Args:
            value: Raw value read from an input field.
            field_name: User-facing field name used in warning messages.
            minimum: Minimum allowed value (inclusive).

        Returns:
            A validated integer, or None if validation fails.
        """
        try:
            parsed_value = int(value)
        except ValueError:
            messagebox.showwarning("Invalid Input", f"{field_name} must be an integer.")
            return None

        if parsed_value < minimum:
            messagebox.showwarning("Invalid Input", f"{field_name} must be at least {minimum}.")
            return None

        return parsed_value

    def _generate_colorpoints_coords(self, width: int, height: int):
        """Generate colorpoint coordinates from the currently selected input mode.

        Args:
            width: Source image width.
            height: Source image height.

        Returns:
            A tuple containing the boolean coordinate array and a short description string.
            If validation fails, returns (None, "").
        """
        mode = self.colorpoint_mode_var.get()

        if mode == "Uniform":
            w_interval = self._read_int(self.uniform_w_interval_var.get(), "Width interval", 0)
            h_interval = self._read_int(self.uniform_h_interval_var.get(), "Height interval", 0)

            if w_interval is None or h_interval is None:
                return None, ""

            colorpoints_coords = uniform_bool_array(width, height, w_interval, h_interval)
            description = f"uniform points, with a width interval of {w_interval} pixels and a \
            height interval of {h_interval} pixels."
            return colorpoints_coords, description

        if mode == "Manual":
            if not self._ensure_manual_selection_ready():
                return None, ""

            number_of_points = self._get_manual_selected_point_count()
            if number_of_points == 0:
                messagebox.showwarning(
                    "No Points Selected",
                    "Select at least one manual point by clicking and dragging on the source image.",
                )
                return None, ""

            description = f"manual points ({number_of_points} points)"
            return self.manual_colorpoints_coords.copy(), description

        number_of_points = self._read_int(self.random_points_var.get(), "number of colorpoints", 1)
        if number_of_points is None:
            return None, ""

        max_points = width * height
        if number_of_points > max_points:
            messagebox.showwarning(
                "Invalid Input",
                f"number of colorpoints must be at most {max_points} for this image size.",
            )
            return None, ""

        colorpoints_coords = random_bool_array(width, height, number_of_points)
        description = f"random points ({number_of_points} colorpoints)"
        return colorpoints_coords, description

    def _set_default_window_size(self) -> None:
        """Set the default window size used when opening the app.

        Returns:
            None.
        """
        self.update_idletasks()
        self.geometry(f"{DEFAULT_WINDOW_WIDTH}x{DEFAULT_WINDOW_HEIGHT}")

    def _bring_to_foreground(self) -> None:
        """Bring the application window to the foreground on startup.

        Returns:
            None.
        """
        self.update_idletasks()
        self.deiconify()
        self.lift()
        self.focus_force()
        self.attributes("-topmost", True)
        self.after(FOREGROUND_DELAY_MS, lambda: self.attributes("-topmost", False))

    def _set_all_placeholders(self) -> None:
        """Set all four subimages to the white placeholder.

        Returns:
            None.
        """
        self._set_subimage("source", None)
        self._set_subimage("greyscale", None)
        self._set_subimage("mixed", None)
        self._set_subimage("recolorised", None)

    def _on_image_grid_resize(self, event) -> None:
        """Update frame margins and panel gaps based on current image grid size.

        Args:
            event: Tkinter configure event for the image grid.

        Returns:
            None.
        """
        width = max(event.width, 1)
        height = max(event.height, 1)

        frame_margin_x = max(MIN_FRAME_MARGIN, int(width * FRAME_MARGIN_RATIO))
        frame_margin_y = max(MIN_FRAME_MARGIN, int(height * FRAME_MARGIN_RATIO))
        panel_gap_x = max(MIN_PANEL_GAP, int(width * PANEL_GAP_RATIO))
        panel_gap_y = max(MIN_PANEL_GAP, int(height * PANEL_GAP_RATIO))

        self.image_grid.configure(padx=frame_margin_x, pady=frame_margin_y)
        for panel in self.panels.values():
            panel.grid_configure(padx=panel_gap_x, pady=panel_gap_y)
        self.frame_margin_x = frame_margin_x
        self.frame_margin_y = frame_margin_y
        self.panel_gap_x = panel_gap_x
        self.panel_gap_y = panel_gap_y
        self._update_panel_sizes()

    def _fit_size_to_aspect(self, width: int, height: int, aspect_ratio: float) -> tuple[int, int]:
        """Fit a width-height box to a target aspect ratio.

        Args:
            width: Maximum available width.
            height: Maximum available height.
            aspect_ratio: Target width / height ratio.

        Returns:
            The fitted width and height.
        """
        width = max(width, 1)
        height = max(height, 1)

        if width / height > aspect_ratio:
            fitted_height = height
            fitted_width = max(1, int(round(fitted_height * aspect_ratio)))
        else:
            fitted_width = width
            fitted_height = max(1, int(round(fitted_width / aspect_ratio)))

        return fitted_width, fitted_height

    def _update_panel_sizes(self) -> None:
        """Resize all subimage panels to match the active image aspect ratio.

        Returns:
            None.
        """
        grid_width = max(self.image_grid.winfo_width(), 1)
        grid_height = max(self.image_grid.winfo_height(), 1)

        usable_width = max(1, grid_width - (2 * self.frame_margin_x))
        usable_height = max(1, grid_height - (2 * self.frame_margin_y))
        cell_width = max(1, int(usable_width / 2) - (2 * self.panel_gap_x))
        cell_height = max(1, int(usable_height / 2) - (2 * self.panel_gap_y))

        panel_width, panel_height = self._fit_size_to_aspect(
            cell_width,
            cell_height,
            self.panel_aspect_ratio,
        )

        for panel in self.panels.values():
            panel.configure(width=panel_width, height=panel_height)

    def _on_panel_resize(self, key: str) -> None:
        """Redraw a panel after its size changes.

        Args:
            key: Panel identifier.

        Returns:
            None.
        """
        self._set_subimage(key, self._panel_images.get(key))

    def _prepare_display_image(
        self,
        image: Image.Image,
        panel_width: int,
        panel_height: int,
    ) -> Image.Image:
        """Resize an image to fit within the panel while keeping aspect ratio.

        Args:
            image: Input image to display.
            panel_width: Current panel width in pixels.
            panel_height: Current panel height in pixels.

        Returns:
            The resized image that fits inside the panel.
        """
        preview = image.copy().convert("RGB")
        render_width = max(1, panel_width)
        render_height = max(1, panel_height)
        return preview.resize((render_width, render_height), RESAMPLE_FILTER)

    def _set_subimage(self, key: str, image: Image.Image | None) -> None:
        """Display an image (or placeholder) in one of the 2x2 grid positions.

        Args:
            key: Subimage identifier.
            image: Image to display. If None, the white placeholder is used.

        Returns:
            None.
        """
        self._panel_images[key] = image

        panel = self.panels[key]
        panel.delete("all")
        panel.configure(bg=SUBIMAGE_BG)

        if image is None:
            self._photo_refs.pop(key, None)
            return

        panel_width = max(panel.winfo_width(), 1)
        panel_height = max(panel.winfo_height(), 1)
        render = self._prepare_display_image(image, panel_width, panel_height)
        photo = ImageTk.PhotoImage(render)
        panel.create_image(panel_width // 2, panel_height // 2, image=photo, anchor="center")
        self._photo_refs[key] = photo

    def _require_source_image(self) -> bool:
        """Check whether the source image exists.

        Returns:
            True if the source image is loaded, otherwise False.
        """
        if self.image_source is None:
            messagebox.showwarning("Image Required", "Load a source image first.")
            return False
        return True

    def clear_images(self) -> None:
        """Clear all currently shown images and reset related state.

        Returns:
            None.
        """
        self.image_source = None
        self.image_greyscale = None
        self.image_mixed = None
        self.image_recolorised = None
        self.colorpoints_coords = None
        self.colorpoint_mode_used = None
        self.greyscale_method_used = None
        self._reset_manual_state()
        self.panel_aspect_ratio = CELL_WIDTH / CELL_HEIGHT
        self._update_panel_sizes()
        self._set_all_placeholders()
        self._update_uniform_pointcount_text()
        self.status_var.set("Cleared all images.")

    def load_source_image(self) -> None:
        """Load the source image from the filesystem.

        Returns:
            None.
        """
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif"),
                ("All Files", "*.*"),
            ],
        )

        if not file_path:
            return

        try:
            loaded = Image.open(file_path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not load image.\n\n{exc}")
            return

        self.image_source = loaded
        self.panel_aspect_ratio = loaded.width / loaded.height
        self._update_panel_sizes()
        self.image_greyscale = None
        self.image_mixed = None
        self.image_recolorised = None
        self.colorpoints_coords = None
        self.colorpoint_mode_used = None
        self.greyscale_method_used = None
        self._reset_manual_state()

        self._set_subimage("source", self.image_source)
        self._set_subimage("greyscale", None)
        self._set_subimage("mixed", None)
        self._set_subimage("recolorised", None)
        self._update_uniform_pointcount_text()
        self.status_var.set("Loaded source image.")

    def create_greyscale_image(self) -> None:
        """Generate and display the greyscale image from the source image.

        Returns:
            None.
        """
        if not self._require_source_image():
            return

        greyscale_method = self._get_selected_greyscale_method_code()
        self.image_greyscale = RGB_to_greyscale(self.image_source, greyscale_method)
        self.greyscale_method_used = greyscale_method
        self._set_subimage("greyscale", self.image_greyscale)
        self.status_var.set(f"Generated greyscale image using method: {greyscale_method}.")

    def create_mixed_image(self) -> None:
        """Generate and display the mixed image using selected colorpoint inputs.

        Returns:
            None.
        """
        if not self._require_source_image():
            return

        width, height = self.image_source.size
        colorpoints_coords, description = self._generate_colorpoints_coords(width, height)
        if colorpoints_coords is None:
            return

        self.colorpoints_coords = colorpoints_coords
        self.colorpoint_mode_used = self.colorpoint_mode_var.get()

        greyscale_method = self._get_selected_greyscale_method_code()
        self.image_mixed = generate_mixed_img(
            self.image_source,
            self.colorpoints_coords,
            greyscale_method,
        )
        self._set_subimage("mixed", self.image_mixed)
        self.status_var.set(f"Generated mixed image using {description}.")

    def create_recolorised_image(self) -> None:
        """Generate and display the recolorised image from the greyscale image.

        Returns:
            None.
        """
        if not self._require_source_image():
            return

        greyscale_method = self._get_selected_greyscale_method_code()
        if self.image_greyscale is None or self.greyscale_method_used != greyscale_method:
            self.image_greyscale = RGB_to_greyscale(self.image_source, greyscale_method)
            self.greyscale_method_used = greyscale_method
            self._set_subimage("greyscale", self.image_greyscale)

        current_mode = self.colorpoint_mode_var.get()
        if self.colorpoints_coords is None or self.colorpoint_mode_used != current_mode:
            width, height = self.image_source.size
            colorpoints_coords, _description = self._generate_colorpoints_coords(width, height)
            if colorpoints_coords is None:
                return

            self.colorpoints_coords = colorpoints_coords
            self.colorpoint_mode_used = current_mode

        D = get_D(self.image_source, self.colorpoints_coords)
        self.image_recolorised = recolorise(D, self.image_greyscale)
        self._set_subimage("recolorised", self.image_recolorised)
        self.status_var.set("Generated recolorised image.")


if __name__ == "__main__":
    app = RecolorizerApp()
    app.mainloop()
