import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
        self.greyscale_method_used = None
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
        self._create_button(button_column, "Generate Mixed Image", self.create_mixed_image)
        self._create_button(button_column, "Generate Recolorised", self.create_recolorised_image)

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

    def _create_button(self, parent: tk.Widget, text: str, command, side: str = "top") -> None:
        """Create one control button in the right-hand column.

        Args:
            parent: Parent widget for the button.
            text: Text shown on the button.
            command: Callback function triggered on button press.
            side: Pack side for placing the button.

        Returns:
            None.
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
            values=("Random", "Uniform"),
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

        self.colorpoint_mode_var.trace_add("write", self._on_colorpoint_mode_change)
        self.uniform_w_interval_var.trace_add("write", self._on_uniform_interval_change)
        self.uniform_h_interval_var.trace_add("write", self._on_uniform_interval_change)
        self._update_uniform_pointcount_text()
        self._update_colorpoint_input_fields()

    def _on_colorpoint_mode_change(self, *_args) -> None:
        """Handle updates when the colorpoint mode dropdown value changes.

        Args:
            *_args: Tkinter trace arguments.

        Returns:
            None.
        """
        self._update_colorpoint_input_fields()

    def _on_uniform_interval_change(self, *_args) -> None:
        """Handle updates when the uniform interval inputs are edited.

        Args:
            *_args: Tkinter trace arguments.

        Returns:
            None.
        """
        self._update_uniform_pointcount_text()

    def _update_colorpoint_input_fields(self) -> None:
        """Show only the input fields relevant to the selected colorpoint mode.

        Returns:
            None.
        """
        mode = self.colorpoint_mode_var.get()
        self.random_input_frame.pack_forget()
        self.uniform_input_frame.pack_forget()

        if mode == "Uniform":
            self.uniform_input_frame.pack(fill="x")
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
        self.greyscale_method_used = None
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
        self.greyscale_method_used = None

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

        if self.colorpoints_coords is None:
            width, height = self.image_source.size
            colorpoints_coords, _description = self._generate_colorpoints_coords(width, height)
            if colorpoints_coords is None:
                return

            self.colorpoints_coords = colorpoints_coords

        D = get_D(self.image_source, self.colorpoints_coords)
        self.image_recolorised = recolorise(D, self.image_greyscale)
        self._set_subimage("recolorised", self.image_recolorised)
        self.status_var.set("Generated recolorised image.")


if __name__ == "__main__":
    app = RecolorizerApp()
    app.mainloop()
