import tkinter as tk
from tkinter import filedialog, Scale, Frame, IntVar, Checkbutton, Button, Radiobutton, messagebox
from tkinter import ttk  # Import ttk for tabs
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
from mpl_toolkits.mplot3d import Axes3D  # For 3D visualization
import seaborn as sns  # New import for seaborn

# Set the seaborn style globally
sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)

class ConcretePoreAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Concrete Pore Analyzer")
        self.root.geometry("1400x900")
        
        # Variables
        self.original_image = None
        self.current_image = None
        self.pore_mask = None
        self.pore_percentage = 0
        self.show_contours = IntVar(value=1)
        self.show_stats = IntVar(value=1)
        
        # Sample selection variables
        self.sample_mode = False
        self.sample_type = "pore"  # "pore" or "concrete"
        self.sample_pore_color = None
        self.sample_concrete_color = None
        
        # ROI selection variables
        self.roi_selection_mode = False
        self.roi_start_point = None
        self.roi_end_point = None
        self.roi_rectangle = None
        self.roi_mask = None
        self.analyze_roi_only = IntVar(value=0)
        
        # Add variables for edge detection
        self.use_edge_detection = IntVar(value=0)
        self.edge_low_threshold = IntVar(value=50)
        self.edge_high_threshold = IntVar(value=150)
        self.edge_combine_mode = IntVar(value=0)  # 0=enhance, 1=filter, 2=standalone
        
        # Scale selection variables
        self.scale_mode = False
        self.scale_line = None  # (x1, y1, x2, y2) in original image coords
        self.scale_length_mm = None  # Changed from scale_length_cm
        self.mm_per_px = None  # Changed from cm_per_px - None if not set
        
        # Image file information
        self.image_filename = None
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a horizontal layout with parameters left and images right
        # Left panel for controls
        left_panel = Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=10, pady=10)
        
        # Right panel for images and histogram
        right_panel = Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook (tabs) for controls in left panel
        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.main_tab = Frame(self.notebook)
        self.parameters_tab = Frame(self.notebook)
        self.edge_detection_tab = Frame(self.notebook)
        self.sample_tab = Frame(self.notebook)
        self.roi_tab = Frame(self.notebook)
        self.scale_tab = Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.main_tab, text="Main")
        self.notebook.add(self.parameters_tab, text="Parameters")
        self.notebook.add(self.edge_detection_tab, text="Edge Detection")
        self.notebook.add(self.sample_tab, text="Sample Selection")
        self.notebook.add(self.roi_tab, text="ROI")
        self.notebook.add(self.scale_tab, text="Scale Selection")
        
        # Images display frame in right panel (top)
        images_frame = Frame(right_panel)
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original image display
        original_frame = Frame(images_frame)
        original_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        tk.Label(original_frame, text="Original Image").pack()
        self.original_image_label = tk.Label(original_frame, bg="lightgray")
        self.original_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Processed image display
        processed_frame = Frame(images_frame)
        processed_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        tk.Label(processed_frame, text="Pore Detection Result").pack()
        self.processed_image_label = tk.Label(processed_frame, bg="lightgray")
        self.processed_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Pure mask display
        mask_frame = Frame(images_frame)
        mask_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        tk.Label(mask_frame, text="Pore Mask").pack()
        self.mask_image_label = tk.Label(mask_frame, bg="lightgray")
        self.mask_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights to make images expand properly
        images_frame.grid_columnconfigure(0, weight=1)
        images_frame.grid_columnconfigure(1, weight=1)
        images_frame.grid_rowconfigure(0, weight=1)
        images_frame.grid_rowconfigure(1, weight=1)
        
        # Histogram frame (below images in right panel)
        self.histogram_frame = Frame(right_panel)
        self.histogram_frame.pack(fill=tk.X, expand=False, pady=5)
        
        # ===== TAB 1: MAIN =====
        # Main buttons and results
        btn_frame = Frame(self.main_tab)
        btn_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(btn_frame, text="Load Image", command=self.load_image, width=15).pack(pady=5)
        tk.Button(btn_frame, text="Save Mask", command=self.save_mask, width=15).pack(pady=5)
        tk.Button(btn_frame, text="Save Analysis", command=self.save_analysis, width=15).pack(pady=5)
        
        # Add Finish Analysis button
        tk.Button(btn_frame, text="Finish Analysis", command=self.finish_analysis,
                 bg="green", fg="white", font=("Arial", 10, "bold"), width=15).pack(pady=5)
        
        # Add a prominent reset button
        tk.Button(btn_frame, text="Reset All", command=self.reset_all,
                 bg="red", fg="white", font=("Arial", 10, "bold"), width=15).pack(pady=(15,5))
        
        # Options
        options_frame = Frame(self.main_tab)
        options_frame.pack(fill=tk.X, pady=10)
        
        Checkbutton(options_frame, text="Show Contours", variable=self.show_contours, 
                   command=self.update_image).pack(anchor=tk.W)
        Checkbutton(options_frame, text="Show Statistics", variable=self.show_stats, 
                   command=self.update_image).pack(anchor=tk.W)
        
        # Stats display
        self.stats_frame = Frame(self.main_tab)
        self.stats_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(self.stats_frame, text="Analysis Results", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.pore_percentage_label = tk.Label(self.stats_frame, text="Porosity: 0.00%")
        self.pore_percentage_label.pack(anchor=tk.W)
        self.pore_count_label = tk.Label(self.stats_frame, text="Pore Count: 0")
        self.pore_count_label.pack(anchor=tk.W)
        self.avg_size_label = tk.Label(self.stats_frame, text="Avg Pore Size: 0.00 px²")
        self.avg_size_label.pack(anchor=tk.W)
        self.analyzed_area_label = tk.Label(self.stats_frame, text="Analyzed Area: Full Image")
        self.analyzed_area_label.pack(anchor=tk.W)
        
        # ===== TAB 2: PARAMETERS =====
        # Controls frame with scrolling
        param_canvas = tk.Canvas(self.parameters_tab, width=250)
        param_scrollbar = tk.Scrollbar(self.parameters_tab, orient="vertical", command=param_canvas.yview)
        
        # Configure the canvas
        param_canvas.configure(yscrollcommand=param_scrollbar.set)
        param_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        param_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create inner frame for controls
        controls_frame = Frame(param_canvas)
        param_canvas.create_window((0, 0), window=controls_frame, anchor="nw")
        
        # Configure scrolling for the canvas
        def configure_scroll_region(event):
            param_canvas.configure(scrollregion=param_canvas.bbox("all"))
        controls_frame.bind("<Configure>", configure_scroll_region)
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            param_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        param_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        tk.Label(controls_frame, text="Pre-processing Controls", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        # Gaussian blur control
        tk.Label(controls_frame, text="Blur Kernel:").pack(anchor=tk.W)
        blur_frame = Frame(controls_frame)
        blur_frame.pack(fill=tk.X, pady=5)
        self.blur_scale = Scale(blur_frame, from_=1, to=21, resolution=2, orient=tk.HORIZONTAL,
                               command=self.update_image)
        self.blur_scale.set(5)
        self.blur_scale.pack(fill=tk.X)
        
        # Adaptive threshold controls
        tk.Label(controls_frame, text="Block Size:").pack(anchor=tk.W)
        block_frame = Frame(controls_frame)
        block_frame.pack(fill=tk.X, pady=5)
        
        # Add the missing block_scale control here
        self.block_scale = Scale(block_frame, from_=3, to=51, resolution=2, orient=tk.HORIZONTAL,
                               command=self.update_image)
        self.block_scale.set(11)  # Default block size
        self.block_scale.pack(fill=tk.X)

        tk.Label(controls_frame, text="C Value:").pack(anchor=tk.W)
        c_frame = Frame(controls_frame)
        c_frame.pack(fill=tk.X, pady=5)
        self.c_scale = Scale(c_frame, from_=-30, to=30, orient=tk.HORIZONTAL,
                           command=self.update_image)
        self.c_scale.set(2)
        self.c_scale.pack(fill=tk.X)
        
        # Morphological operations
        tk.Label(controls_frame, text="Morphological Operation Size:").pack(anchor=tk.W)
        morph_frame = Frame(controls_frame)
        morph_frame.pack(fill=tk.X, pady=5)
        self.morph_scale = Scale(morph_frame, from_=0, to=10, orient=tk.HORIZONTAL,
                               command=self.update_image)
        self.morph_scale.set(1)
        self.morph_scale.pack(fill=tk.X)
        
        # Pore size range controls
        tk.Label(controls_frame, text="Pore Size Range:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(10,0))
        
        # Min pore size
        tk.Label(controls_frame, text="Min Pore Size (px²):").pack(anchor=tk.W)
        minsize_frame = Frame(controls_frame)
        minsize_frame.pack(fill=tk.X, pady=5)
        self.min_size_scale = Scale(minsize_frame, from_=0, to=1000, orient=tk.HORIZONTAL,
                                  command=self.update_image)
        self.min_size_scale.set(30)
        self.min_size_scale.pack(fill=tk.X)
        
        # Max pore size
        tk.Label(controls_frame, text="Max Pore Size (px²):").pack(anchor=tk.W)
        maxsize_frame = Frame(controls_frame)
        maxsize_frame.pack(fill=tk.X, pady=5)
        self.max_size_scale = Scale(maxsize_frame, from_=1000, to=100000, orient=tk.HORIZONTAL,
                                  command=self.update_image)
        self.max_size_scale.set(100000)  # Default to a large value
        self.max_size_scale.pack(fill=tk.X)
        
        # ===== TAB 3: EDGE DETECTION =====
        tk.Label(self.edge_detection_tab, text="Edge Detection", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=10)
        
        Checkbutton(self.edge_detection_tab, text="Use Canny Edge Detection", variable=self.use_edge_detection, 
                   command=self.update_image).pack(anchor=tk.W)
        
        # Edge detection parameters
        edge_params_frame = Frame(self.edge_detection_tab)
        edge_params_frame.pack(fill=tk.X, pady=5)
        
        # Low threshold
        tk.Label(edge_params_frame, text="Low Threshold:").pack(anchor=tk.W)
        low_thresh_frame = Frame(edge_params_frame)
        low_thresh_frame.pack(fill=tk.X, pady=2)
        Scale(low_thresh_frame, from_=1, to=255, orient=tk.HORIZONTAL,
             variable=self.edge_low_threshold, command=self.update_image).pack(fill=tk.X)
        
        # High threshold
        tk.Label(edge_params_frame, text="High Threshold:").pack(anchor=tk.W)
        high_thresh_frame = Frame(edge_params_frame)
        high_thresh_frame.pack(fill=tk.X, pady=2)
        Scale(high_thresh_frame, from_=1, to=255, orient=tk.HORIZONTAL,
             variable=self.edge_high_threshold, command=self.update_image).pack(fill=tk.X)
        
        # Edge detection mode
        edge_mode_frame = Frame(self.edge_detection_tab)
        edge_mode_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(edge_mode_frame, text="Edge Detection Mode:").pack(anchor=tk.W)
        Radiobutton(edge_mode_frame, text="Enhance Thresholding", variable=self.edge_combine_mode, 
                   value=0, command=self.update_image).pack(anchor=tk.W)
        Radiobutton(edge_mode_frame, text="Filter Pore Boundaries", variable=self.edge_combine_mode, 
                   value=1, command=self.update_image).pack(anchor=tk.W)
        Radiobutton(edge_mode_frame, text="Standalone Edge Mode", variable=self.edge_combine_mode, 
                   value=2, command=self.update_image).pack(anchor=tk.W)
        
        # ===== TAB 4: SAMPLE SELECTION =====
        sample_frame = Frame(self.sample_tab)
        sample_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(sample_frame, text="Sample Selection", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        tk.Label(sample_frame, text="⚠️ Warning: Sample selection changes detection method!", 
                 font=("Arial", 9), fg="red").pack(anchor=tk.W, pady=2)
        
        Button(sample_frame, text="Select Sample Pore", command=lambda: self.start_sample_selection("pore"), 
               bg="lightblue", width=15).pack(pady=5)
        Button(sample_frame, text="Select Concrete", command=lambda: self.start_sample_selection("concrete"), 
               bg="lightgrey", width=15).pack(pady=5)
        Button(sample_frame, text="Reset Samples", command=self.reset_samples, 
               bg="pink", width=15).pack(pady=5)
        Button(sample_frame, text="Reset to Adaptive Threshold", command=self.reset_detection_method, 
               bg="orange", width=20).pack(pady=5)
        
        # Sample Status
        self.sample_status_frame = Frame(self.sample_tab)
        self.sample_status_frame.pack(fill=tk.X, pady=5)
        self.pore_sample_label = tk.Label(self.sample_status_frame, text="Pore Sample: Not Selected", fg="red")
        self.pore_sample_label.pack(anchor=tk.W)
        self.concrete_sample_label = tk.Label(self.sample_status_frame, text="Concrete Sample: Not Selected", fg="red")
        self.concrete_sample_label.pack(anchor=tk.W)
        
        # Color similarity threshold
        tk.Label(self.sample_tab, text="Pore Color Similarity Threshold:").pack(anchor=tk.W)
        sim_frame = Frame(self.sample_tab)
        sim_frame.pack(fill=tk.X, pady=5)
        self.similarity_scale = Scale(sim_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                    command=self.update_image)
        self.similarity_scale.set(20)  # Default similarity threshold
        self.similarity_scale.pack(fill=tk.X)
        
        # Add new concrete similarity threshold
        tk.Label(self.sample_tab, text="Concrete Color Similarity Threshold:").pack(anchor=tk.W)
        conc_sim_frame = Frame(self.sample_tab)
        conc_sim_frame.pack(fill=tk.X, pady=5)
        self.concrete_similarity_scale = Scale(conc_sim_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                        command=self.update_image)
        self.concrete_similarity_scale.set(20)  # Default concrete similarity threshold
        self.concrete_similarity_scale.pack(fill=tk.X)
        
        # ===== TAB 5: ROI =====
        roi_frame = Frame(self.roi_tab)
        roi_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(roi_frame, text="Region of Interest", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        Button(roi_frame, text="Select ROI", command=self.start_roi_selection, 
               bg="lightgreen", width=15).pack(pady=5)
        Button(roi_frame, text="Clear ROI", command=self.clear_roi, 
               bg="pink", width=15).pack(pady=5)
        Checkbutton(roi_frame, text="Analyze ROI Only", variable=self.analyze_roi_only, 
                   command=self.update_image).pack(anchor=tk.W, pady=5)
        
        self.roi_status_label = tk.Label(roi_frame, text="ROI: Not Selected", fg="red")
        self.roi_status_label.pack(anchor=tk.W)
        
        # Bind mouse events for sample selection and ROI selection
        self.original_image_label.bind("<Button-1>", self.on_mouse_down)
        self.original_image_label.bind("<B1-Motion>", self.on_mouse_drag)
        self.original_image_label.bind("<ButtonRelease-1>", self.on_mouse_up)

        # ===== TAB 6: ADVANCED ANALYSIS =====
        # Create Advanced Analysis tab
        self.advanced_analysis_tab = Frame(self.notebook)
        self.notebook.add(self.advanced_analysis_tab, text="Advanced Analysis")

        # Title for advanced analysis tab
        tk.Label(self.advanced_analysis_tab, text="Advanced Pore Analysis", 
                font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=10, padx=10)

        # Create button frame
        analysis_buttons_frame = Frame(self.advanced_analysis_tab)
        analysis_buttons_frame.pack(fill=tk.X, padx=10, pady=5)

        # Add analysis buttons with descriptive text
        shape_frame = Frame(self.advanced_analysis_tab)
        shape_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(shape_frame, text="1. Shape Analysis:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        tk.Label(shape_frame, text="   Shape factor distribution, shape vs area,\n   circularity assessment", 
                 font=("Arial", 9)).pack(anchor=tk.W)
        tk.Button(shape_frame, text="Shape Factor Analysis", 
                 command=lambda: self.show_shape_analysis(),
                 bg="lightblue", width=20).pack(pady=5)

        size_frame = Frame(self.advanced_analysis_tab)
        size_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(size_frame, text="2. Size Analysis:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        tk.Label(size_frame, text="   Size distributions, pore diameter,\n   cumulative distribution", 
                 font=("Arial", 9)).pack(anchor=tk.W)
        tk.Button(size_frame, text="Pore Size Analysis", 
                 command=lambda: self.show_size_analysis(),
                 bg="lightgreen", width=20).pack(pady=5)

        porosity_frame = Frame(self.advanced_analysis_tab)
        porosity_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(porosity_frame, text="3. Porosity Analysis:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        tk.Label(porosity_frame, text="   Spatial distribution, 3D map,\n   porosity assessment", 
                 font=("Arial", 9)).pack(anchor=tk.W)
        tk.Button(porosity_frame, text="Porosity Analysis", 
                 command=lambda: self.show_porosity_analysis(),
                 bg="lightyellow", width=20).pack(pady=5)

        # Container for displaying analysis results status
        self.analysis_status = tk.Label(self.advanced_analysis_tab, text="Select an analysis type above", fg="blue")
        self.analysis_status.pack(pady=10)

        # ===== TAB 6: SCALE SELECTION =====
        scale_frame = Frame(self.scale_tab)
        scale_frame.pack(fill=tk.X, pady=10)

        tk.Label(scale_frame, text="Scale Selection", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        tk.Label(scale_frame, text="Right-click and drag to draw scale line", font=("Arial", 9)).pack(anchor=tk.W, pady=5)

        Button(scale_frame, text="Draw Scale Line", command=self.start_scale_selection, 
               bg="lightyellow", width=15).pack(pady=5)
        Button(scale_frame, text="Clear Scale", command=self.clear_scale, 
               bg="pink", width=15).pack(pady=5)

        self.scale_status_label = tk.Label(scale_frame, text="Scale: Not Set", fg="red")
        self.scale_status_label.pack(anchor=tk.W, pady=5)

        # Instructions
        instructions_frame = Frame(self.scale_tab)
        instructions_frame.pack(fill=tk.X, pady=10, padx=5)
        tk.Label(instructions_frame, text="Instructions:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        tk.Label(instructions_frame, text="1. Click 'Draw Scale Line'", font=("Arial", 8)).pack(anchor=tk.W)
        tk.Label(instructions_frame, text="2. Right-click and drag on image", font=("Arial", 8)).pack(anchor=tk.W)
        tk.Label(instructions_frame, text="3. Enter real-world length in mm", font=("Arial", 8)).pack(anchor=tk.W)  # Changed from cm
        tk.Label(instructions_frame, text="4. All results will show in mm", font=("Arial", 8)).pack(anchor=tk.W)  # Changed from cm

        # Bind mouse events for scale selection
        self.original_image_label.bind("<Button-3>", self.on_scale_mouse_down)  # Right click
        self.original_image_label.bind("<B3-Motion>", self.on_scale_mouse_drag)
        self.original_image_label.bind("<ButtonRelease-3>", self.on_scale_mouse_up)

    def on_mouse_down(self, event):
        """Handle mouse button down event for both sample selection and ROI"""
        if self.original_image is None:
            return
            
        if self.sample_mode:
            # Handle sample selection (single click)
            self.on_image_click(event)
        elif self.roi_selection_mode:
            # Convert display coordinates to original image coordinates
            x_orig, y_orig = self.display_to_original_coords(event.x, event.y)
            
            # If click was outside the image, ignore
            if x_orig is None or y_orig is None:
                return
            
            self.roi_start_point = (x_orig, y_orig)

    def on_mouse_drag(self, event):
        """Handle mouse drag for ROI selection"""
        if not self.roi_selection_mode or self.roi_start_point is None:
            return
    
        # Convert display coordinates to original image coordinates
        x_orig, y_orig = self.display_to_original_coords(event.x, event.y)
    
        # If drag is outside the image, ignore
        if x_orig is None or y_orig is None:
            return
        
        self.roi_end_point = (x_orig, y_orig)
        
        # Create a temporary image for display with the current ROI rectangle
        display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB).copy()
        
        # Draw the temporary ROI rectangle on the display
        x1 = min(self.roi_start_point[0], self.roi_end_point[0])
        y1 = min(self.roi_start_point[1], self.roi_end_point[1])
        x2 = max(self.roi_start_point[0], self.roi_end_point[0])
        y2 = max(self.roi_start_point[1], self.roi_end_point[1])
        
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow rectangle
        
        # Calculate temporary ROI dimensions for status display
        width = x2 - x1
        height = y2 - y1
        area = width * height
        self.roi_status_label.config(text=f"ROI: Selecting ({width}×{height} = {area} px²)", fg="blue")
        
        # Get current displayed image dimensions for consistent resizing
        displayed_img = self.original_image_label.image
        disp_width = displayed_img.width()
        disp_height = displayed_img.height()
    
        # Convert to PIL and maintain exact same size as current display
        display_pil = Image.fromarray(display_image)
        display_pil = display_pil.resize((disp_width, disp_height), Image.LANCZOS)
    
        # Update the display without changing size or shape
        display_photo = ImageTk.PhotoImage(display_pil)
        self.original_image_label.config(image=display_photo)
        self.original_image_label.image = display_photo

    def on_mouse_up(self, event):
        """Handle mouse button release for ROI selection"""
        if not self.roi_selection_mode or self.roi_start_point is None:
            return
        
        # Convert display coordinates to original image coordinates
        x_orig, y_orig = self.display_to_original_coords(event.x, event.y)
        
        # If release was outside the image, ignore
        if x_orig is None or y_orig is None:
            return
            
        self.roi_end_point = (x_orig, y_orig)
        
        # Create ROI rectangle in proper order (top-left, bottom-right)
        x1 = min(self.roi_start_point[0], self.roi_end_point[0])
        y1 = min(self.roi_start_point[1], self.roi_end_point[1])
        x2 = max(self.roi_start_point[0], self.roi_end_point[0])
        y2 = max(self.roi_start_point[1], self.roi_end_point[1])
        
        # Validate the ROI (ensure it has some minimum size)
        min_size = 10  # At least 10x10 pixels
        if x2 - x1 < min_size or y2 - y1 < min_size:
            # ROI too small, ignore
            self.roi_start_point = None
            self.roi_end_point = None
            self.roi_rectangle = None
            return
            
        self.roi_rectangle = (x1, y1, x2, y2)
        
        # Create ROI mask
        self.roi_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        self.roi_mask[y1:y2, x1:x2] = 255;
        
        # Update ROI status
        width = x2 - x1
        height = y2 - y1
        area = width * height
        self.roi_status_label.config(text=f"ROI: Selected ({width}×{height} = {area} px²)", fg="green")
        
        # Exit ROI selection mode
        self.roi_selection_mode = False
        self.root.title("Concrete Pore Analyzer")
        
        # Turn on analyze ROI only
        self.analyze_roi_only.set(1)
        
        # Update image with new ROI
        self.update_image()

    def start_sample_selection(self, sample_type):
        """Start sample selection mode"""
        self.sample_mode = True
        self.sample_type = sample_type
        self.roi_selection_mode = False
        
        if sample_type == "pore":
            self.root.title("Concrete Pore Analyzer - SELECT A PORE (Click on image)")
        else:
            self.root.title("Concrete Pore Analyzer - SELECT CONCRETE SURFACE (Click on image)")
    
    def start_roi_selection(self):
        """Start ROI selection mode"""
        if self.original_image is None:
            return
            
        self.roi_selection_mode = True
        self.sample_mode = False
        self.roi_start_point = None
        self.roi_end_point = None
        self.root.title("Concrete Pore Analyzer - SELECT REGION OF INTEREST (Drag on image)")
    
    def clear_roi(self):
        """Clear the selected ROI"""
        self.roi_rectangle = None
        self.roi_mask = None
        self.roi_start_point = None
        self.roi_end_point = None
        self.roi_status_label.config(text="ROI: Not Selected", fg="red")
        self.analyze_roi_only.set(0)
        self.update_image()
    
    def on_image_click(self, event):
        """Handle mouse click on image for sample selection"""
        if not self.sample_mode or self.original_image is None:
            return
        
        # Convert display coordinates to original image coordinates
        x_orig, y_orig = self.display_to_original_coords(event.x, event.y)
        
        # If click was outside the image, ignore
        if x_orig is None or y_orig is None:
            return
        
        # Get color at clicked point
        if self.sample_type == "pore":
            self.sample_pore_color = self.original_image[y_orig, x_orig].copy()
            self.pore_sample_label.config(text=f"Pore Sample: Selected BGR={self.sample_pore_color}", fg="green")
        else:
            self.sample_concrete_color = self.original_image[y_orig, x_orig].copy()
            self.concrete_sample_label.config(text=f"Concrete Sample: Selected BGR={self.sample_concrete_color}", fg="green")
        
        # Exit sample selection mode
        self.sample_mode = False
        self.root.title("Concrete Pore Analyzer")
        
        # Update image with new sample
        self.update_image()
    
    def reset_samples(self):
        """Reset sample selections"""
        self.sample_pore_color = None
        self.sample_concrete_color = None
        self.pore_sample_label.config(text="Pore Sample: Not Selected", fg="red")
        self.concrete_sample_label.config(text="Concrete Sample: Not Selected", fg="red")
        self.sample_mode = False
        self.update_image()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        if file_path:
            # Store the filename without extension
            import os
            self.image_filename = os.path.splitext(os.path.basename(file_path))[0]
            
            self.original_image = cv2.imread(file_path)
            # Reset samples and ROI when loading new image
            self.reset_samples()
            self.clear_roi()
            self.update_image()
            
    def update_image(self, *args):
        if self.original_image is None:
            return
            
        # Display original image with ROI rectangle
        original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        display_original = original_rgb.copy()
        
        # Pre-processing
        # 1. Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply Gaussian blur
        blur_size = self.blur_scale.get()
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # Determine which detection method to use and create binary mask
        detection_method = "Adaptive Thresholding"  # Default
        
        if self.use_edge_detection.get():
            # Edge detection mode
            detection_method = "Edge Detection"
            low_threshold = self.edge_low_threshold.get()
            high_threshold = self.edge_high_threshold.get()
            edges = cv2.Canny(blurred, low_threshold, high_threshold)
            
            if self.edge_combine_mode.get() == 2:  # Standalone mode
                binary = edges
            else:
                # Combine with other methods
                if self.sample_pore_color is not None:
                    detection_method = "Edge + Sample-based"
                    base_binary = self.create_sample_based_mask()
                else:
                    detection_method = "Edge + Adaptive Thresholding"
                    block_size = self.block_scale.get()
                    c_value = self.c_scale.get()
                    base_binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY_INV, block_size, c_value)
                
                # Combine with edges
                if self.edge_combine_mode.get() == 0:  # Enhance mode
                    binary = cv2.bitwise_or(base_binary, edges)
                else:  # Filter mode
                    kernel = np.ones((3, 3), np.uint8)
                    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                    binary = cv2.bitwise_and(base_binary, dilated_edges)
                
        elif self.sample_pore_color is not None:
            # Sample-based detection
            detection_method = "Sample-based Color Detection"
            binary = self.create_sample_based_mask()
        else:
            # Default adaptive thresholding
            detection_method = "Adaptive Thresholding"
            block_size = self.block_scale.get()
            c_value = self.c_scale.get()
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, block_size, c_value)
        
        # 4. Apply morphological operations
        kernel_size = self.morph_scale.get()
        if (kernel_size > 0):
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Apply ROI mask if analyzing ROI only
        if self.analyze_roi_only.get() and self.roi_mask is not None:
            binary = cv2.bitwise_and(binary, binary, mask=self.roi_mask)
        
        # 5. Find and filter contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for visualization
        mask = np.zeros_like(binary)
        
        # Filter by size and draw filtered contours
        min_size = self.min_size_scale.get()
        max_size = self.max_size_scale.get()
        filtered_contours = []
        areas = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_size <= area <= max_size:
                filtered_contours.append(contour)
                areas.append(area)
                cv2.drawContours(mask, [contour], 0, 255, -1)
        
        self.pore_mask = mask.copy()
        
        # Draw ROI rectangle on original image for display
        if self.roi_rectangle is not None:
            x1, y1, x2, y2 = self.roi_rectangle
            cv2.rectangle(display_original, (x1, y1), (x2, y2), (255, 255, 0), 8)  # Bright cyan rectangle, thicker
        
        # Draw scale line if set
        if self.scale_line and self.scale_line[0] and self.scale_line[1]:
            cv2.line(display_original, self.scale_line[0], self.scale_line[1], (255, 0, 255), 6)  # Bright magenta line, thicker
        
        # Calculate statistics with appropriate units
        area_unit = self.get_area_unit()
        length_unit = self.get_length_unit()
        
        if self.analyze_roi_only.get() and self.roi_mask is not None:
            total_area = np.sum(self.roi_mask > 0)
            analyzed_area_text = f"Analyzed Area: ROI ({self.area_px2_to_cm2(total_area):.2f} {area_unit})"
        else:
            total_area = gray.shape[0] * gray.shape[1]
            analyzed_area_text = f"Analyzed Area: Full Image ({self.area_px2_to_cm2(total_area):.2f} {area_unit})"
        
        # Add detection method to the analyzed area label
        self.analyzed_area_label.config(text=f"{analyzed_area_text}\nMethod: {detection_method}")
        
        pore_area = np.sum(mask > 0)
        self.pore_percentage = (pore_area / total_area) * 100 if total_area > 0 else 0
        pore_count = len(filtered_contours)
        avg_size = np.mean(areas) if areas else 0
        
        # Update stats labels with appropriate units
        self.pore_percentage_label.config(text=f"Porosity: {self.pore_percentage:.2f}%")
        self.pore_count_label.config(text=f"Pore Count: {pore_count}")
        self.avg_size_label.config(text=f"Avg Pore Size: {self.area_px2_to_cm2(avg_size):.2f} {area_unit}")
        
        # Create clean copy of original for the processed view
        processed_image = original_rgb.copy()
        
        # Create colored mask for pore visualization
        colored_mask = np.zeros_like(processed_image)
        colored_mask[mask > 0] = [255, 0, 0]  # Red for pores
        
        # Show contours if enabled (on the processed image)
        if self.show_contours.get():
            cv2.drawContours(processed_image, filtered_contours, -1, (0, 255, 0), 2)
        
        # Blend mask with original for visualization
        alpha = 0.5
        blended = cv2.addWeighted(processed_image, 1-alpha, colored_mask, alpha, 0)
        
        # Create a colored mask for direct display
        mask_display = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_display[mask > 0] = [255, 255, 255]  # White for pores
        
        # Convert to PIL format for display
        original_pil = Image.fromarray(display_original)
        processed_pil = Image.fromarray(blended)
        mask_pil = Image.fromarray(mask_display)
        
        # Set size for all images (400x400)
        display_size = (400, 400)
        original_pil.thumbnail(display_size, Image.LANCZOS)
        processed_pil.thumbnail(display_size, Image.LANCZOS)
        mask_pil.thumbnail(display_size, Image.LANCZOS)
        
        # Display all images
        original_photo = ImageTk.PhotoImage(original_pil)
        self.original_image_label.config(image=original_photo)
        self.original_image_label.image = original_photo
        
        processed_photo = ImageTk.PhotoImage(processed_pil)
        self.processed_image_label.config(image=processed_photo)
        self.processed_image_label.image = processed_photo
        
        mask_photo = ImageTk.PhotoImage(mask_pil)
        self.mask_image_label.config(image=mask_photo)
        self.mask_image_label.image = mask_photo
        
        # Update histogram if show_stats is enabled
        if self.show_stats.get():
            self.update_histogram(gray, mask)
        else:
            # Clear histogram
            for widget in self.histogram_frame.winfo_children():
                widget.destroy()
    
    def create_sample_based_mask(self):
        """Create binary mask based on color similarity to samples"""
        pore_similarity_threshold = self.similarity_scale.get()
        concrete_similarity_threshold = self.concrete_similarity_scale.get()
        
        # Create empty mask
        mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        
        # If we have a pore sample, use it
        if self.sample_pore_color is not None:
            # Calculate color distance for each pixel from the sample pore color
            color_distances = np.zeros_like(mask, dtype=np.float32)
            
            for i in range(3):  # For each BGR channel
                color_distances += np.square(
                    self.original_image[:,:,i].astype(np.float32) - 
                    self.sample_pore_color[i]
                )
            
            color_distances = np.sqrt(color_distances)
            
            # Create mask based on pore similarity threshold
            pore_mask = (color_distances <= pore_similarity_threshold)
            mask[pore_mask] = 255
            
            # If concrete sample also available, refine mask
            if self.sample_concrete_color is not None:
                concrete_distances = np.zeros_like(mask, dtype=np.float32)
                
                for i in range(3):  # For each BGR channel
                    concrete_distances += np.square(
                        self.original_image[:,:,i].astype(np.float32) - 
                        self.sample_concrete_color[i]
                    )
                
                concrete_distances = np.sqrt(concrete_distances)
                
                # Pixels that are within concrete similarity threshold of the concrete sample
                # should not be considered pores
                concrete_mask = (concrete_distances <= concrete_similarity_threshold)
                mask[concrete_mask] = 0
        
        return mask
        
    def update_histogram(self, gray, mask):
        """Update the histogram display with detailed porosity analysis"""
        # Clear previous histogram
        for widget in self.histogram_frame.winfo_children():
            widget.destroy()
        
        # Create histogram figure
        fig = Figure(figsize=(8, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        # Prepare mask for histogram calculation
        hist_mask = None
        if self.analyze_roi_only.get() and self.roi_mask is not None:
            hist_mask = self.roi_mask
    
        # Calculate histograms
        hist_full = cv2.calcHist([gray], [0], hist_mask, [256], [0, 256])
        hist_pores = cv2.calcHist([gray], [0], mask, [256], [0, 256])
        
        # Plot histograms
        ax.plot(hist_full, 'b', label='Full Image' if hist_mask is None else 'ROI')
        ax.plot(hist_pores, 'r', label='Pores')
        ax.set_xlim([0, 256])
        ax.set_title('Gray Level Histogram')
        ax.set_xlabel('Gray Level')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=self.histogram_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # If no pores detected, show message and return
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            tk.Label(self.histogram_frame, text="No pores detected for analysis", 
                    font=("Arial", 12), fg="red").pack(pady=10)
            return
            
        # Calculate pore statistics
        areas = [cv2.contourArea(contour) for contour in contours]
        total_pixels = np.sum(mask > 0)
        
        # Basic statistics
        min_area = min(areas) if areas else 0
        max_area = max(areas) if areas else 0
        mean_area = np.mean(areas) if areas else 0
        median_area = np.median(areas) if areas else 0
        std_area = np.std(areas) if areas else 0
        
        # Create statistics frame
        stats_frame = Frame(self.histogram_frame)
        stats_frame.pack(fill=tk.X, expand=True, pady=10)
        
        # Left column for basic stats
        basic_stats = Frame(stats_frame)
        basic_stats.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=20)
        
        tk.Label(basic_stats, text="Basic Pore Statistics", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        tk.Label(basic_stats, text=f"Total Pore Count: {len(contours)}").pack(anchor=tk.W)
        tk.Label(basic_stats, text=f"Total Pore Area: {self.area_px2_to_cm2(total_pixels):.2f} {self.get_area_unit()}").pack(anchor=tk.W)
        tk.Label(basic_stats, text=f"Porosity: {self.pore_percentage:.2f}%").pack(anchor=tk.W)
        tk.Label(basic_stats, text=f"Minimum Pore Size: {self.area_px2_to_cm2(min_area):.1f} {self.get_area_unit()}").pack(anchor=tk.W)
        tk.Label(basic_stats, text=f"Maximum Pore Size: {self.area_px2_to_cm2(max_area):.1f} {self.get_area_unit()}").pack(anchor=tk.W)
        tk.Label(basic_stats, text=f"Mean Pore Size: {self.area_px2_to_cm2(mean_area):.1f} {self.get_area_unit()}").pack(anchor=tk.W)
        tk.Label(basic_stats, text=f"Median Pore Size: {self.area_px2_to_cm2(median_area):.1f} {self.get_area_unit()}").pack(anchor=tk.W)
        tk.Label(basic_stats, text=f"Standard Deviation: {self.area_px2_to_cm2(std_area):.1f} {self.get_area_unit()}").pack(anchor=tk.W)
        
        # Right column for size distribution
        distribution_frame = Frame(stats_frame)
        distribution_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=True, padx=20)
        
        tk.Label(distribution_frame, text="Pore Size Distribution", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        # Size distribution analysis
        size_ranges = [0, 10, 50, 100, 500, 1000, 5000, 10000, float('inf')]
        size_counts = [0] * (len(size_ranges) - 1)
        
        for area in areas:
            for i in range(len(size_ranges) - 1):
                if size_ranges[i] <= area < size_ranges[i+1]:
                    size_counts[i] += 1
                    break
        
        # Display size distribution
        for i in range(len(size_counts)):
            if i < len(size_ranges) - 1:
                if size_ranges[i+1] == float('inf'):
                    range_text = f"{self.area_px2_to_cm2(size_ranges[i]):.1f}+ {self.get_area_unit()}"
                else:
                    range_text = f"{self.area_px2_to_cm2(size_ranges[i]):.1f}-{self.area_px2_to_cm2(size_ranges[i+1]):.1f} {self.get_area_unit()}"
            else:
                range_text = f"{self.area_px2_to_cm2(size_ranges[i]):.1f}+ {self.get_area_unit()}"
                
            percentage = (size_counts[i] / len(contours) * 100) if len(contours) > 0 else 0
            tk.Label(distribution_frame, 
                    text=f"{range_text}: {size_counts[i]} pores ({percentage:.1f}%)").pack(anchor=tk.W)

    def save_mask(self):
        if self.pore_mask is None:
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG files", "*.png"),
                                                         ("JPEG files", "*.jpg"),
                                                         ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, self.pore_mask)
    
    def save_analysis(self):
        if self.original_image is None:
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                               filetypes=[("Text files", "*.txt"),
                                                         ("All files", "*.*")])
        if file_path:
            with open(file_path, 'w') as f:
                f.write("Concrete Pore Analysis Results\n")
                f.write("==============================\n\n")
                f.write(f"Porosity: {self.pore_percentage:.2f}%\n")
                f.write(f"Pore Count: {self.pore_count_label.cget('text').split(':')[1].strip()}\n")
                f.write(f"Average Pore Size: {self.avg_size_label.cget('text').split(':')[1].strip()}\n")
                
                # Add scale information if set
                if self.mm_per_px is not None:
                    f.write(f"\nScale Information:\n")
                    px_dist = math.hypot(self.scale_line[1][0]-self.scale_line[0][0], 
                                       self.scale_line[1][1]-self.scale_line[0][1])
                    f.write(f"Scale: {self.scale_length_mm:.2f} mm = {px_dist:.1f} px\n")
                    f.write(f"Scale Factor: {self.mm_per_px:.6f} mm/px\n")
                    f.write(f"Units: All measurements in mm/mm²\n")
                else:
                    f.write(f"\nScale: Not set (measurements in px/px²)\n")
                
                f.write(f"\nAnalysis Parameters:\n")
                
                # Record sample information with separate thresholds
                if self.sample_pore_color is not None:
                    f.write(f"Sample Pore Color (BGR): {self.sample_pore_color}\n")
                    f.write(f"Pore Color Similarity Threshold: {self.similarity_scale.get()}\n")
                if self.sample_concrete_color is not None:
                    f.write(f"Sample Concrete Color (BGR): {self.sample_concrete_color}\n")
                    f.write(f"Concrete Color Similarity Threshold: {self.concrete_similarity_scale.get()}\n")
                
                f.write(f"Blur Kernel Size: {self.blur_scale.get()}\n")
                f.write(f"Block Size: {self.block_scale.get()}\n")
                f.write(f"C Value: {self.c_scale.get()}\n")
                f.write(f"Morphological Operation Size: {self.morph_scale.get()}\n")
                f.write(f"Minimum Pore Size: {self.min_size_scale.get()} px²\n")
                f.write(f"Maximum Pore Size: {self.max_size_scale.get()} px²\n")
                
                # Add edge detection parameters
                f.write(f"Use Edge Detection: {'Yes' if self.use_edge_detection.get() else 'No'}\n")
                if self.use_edge_detection.get():
                    mode_names = ["Enhance", "Filter", "Standalone"]
                    f.write(f"Edge Detection Mode: {mode_names[self.edge_combine_mode.get()]}\n")
                    f.write(f"Edge Low Threshold: {self.edge_low_threshold.get()}\n")
                    f.write(f"Edge High Threshold: {self.edge_high_threshold.get()}\n")

    def analyze_spatial_porosity(self, mask):
        """Analyze spatial distribution of porosity across the image"""
        # If no pores detected, return empty data
        if self.pore_mask is None:
            return None, None, None, None, None
            
        # Determine which region to analyze
        if self.analyze_roi_only.get() and self.roi_mask is not None:
            analyzed_area = self.roi_mask
            if self.roi_rectangle:
                x1, y1, x2, y2 = self.roi_rectangle
                height = y2 - y1
                width = x2 - x1
                full_area = np.zeros_like(self.roi_mask)
                full_area[y1:y2, x1:x2] = 1
            else:
                # Fall back to full mask if rectangle not available
                height, width = self.roi_mask.shape
                full_area = np.ones_like(self.roi_mask)
        else:
            height, width = self.original_image.shape[:2]
            analyzed_area = np.ones((height, width), dtype=np.uint8)
            full_area = np.ones((height, width), dtype=np.uint8)
        
        # Divide the image into a grid (e.g., 10x10 cells)
        grid_size = min(10, max(2, int(min(width, height) / 50)))  # Adaptive grid size
        cell_width = width // grid_size
        cell_height = height // grid_size
        
        # Calculate local porosity for each cell
        local_porosity = np.zeros((grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Define cell region
                if self.analyze_roi_only.get() and self.roi_mask is not None and self.roi_rectangle:
                    x1, y1, _, _ = self.roi_rectangle
                    cell_x1 = x1 + j * cell_width
                    cell_y1 = y1 + i * cell_height
                else:
                    cell_x1 = j * cell_width
                    cell_y1 = i * cell_height
                
                cell_x2 = min(cell_x1 + cell_width, width)
                cell_y2 = min(cell_y1 + cell_height, height)
                
                # Extract the cell region from the mask
                cell_mask_region = mask[cell_y1:cell_y2, cell_x1:cell_x2]
                
                # Apply ROI constraint to cell if needed
                if self.analyze_roi_only.get() and self.roi_mask is not None:
                    cell_roi_region = self.roi_mask[cell_y1:cell_y2, cell_x1:cell_x2]
                    # Only count pixels that are within ROI
                    valid_pixels = np.sum(cell_roi_region > 0)
                    pore_pixels = np.sum((cell_mask_region > 0) & (cell_roi_region > 0))
                else:
                    # Count all pixels in the cell
                    valid_pixels = cell_mask_region.size
                    pore_pixels = np.sum(cell_mask_region > 0)
                
                # Calculate porosity in this cell (as percentage)
                if valid_pixels > 0:
                    local_porosity[i, j] = (pore_pixels / valid_pixels) * 100
                else:
                    local_porosity[i, j] = 0
        
        # Calculate statistics from local porosity
        valid_cells = local_porosity > 0
        if np.any(valid_cells):
            valid_porosity = local_porosity[valid_cells]
            mean_local = np.mean(valid_porosity)
            median_local = np.median(valid_porosity)
            std_dev = np.std(valid_porosity)
            min_local = np.min(valid_porosity)
            max_local = np.max(valid_porosity)
            coef_var = (std_dev / mean_local) * 100 if mean_local > 0 else 0
        else:
            mean_local = median_local = std_dev = min_local = max_local = coef_var = 0
        
        # Calculate global porosity (should match the one in update_image)
        global_porosity = self.pore_percentage
        
        # Determine porosity assessment based on global porosity
        if global_porosity < 2:
            assessment = "Low porosity concrete (very dense)"
        elif global_porosity < 5:
            assessment = "Medium-low porosity concrete"
        elif global_porosity < 10:
            assessment = "Medium porosity concrete"
        elif global_porosity < 15:
            assessment = "Medium-high porosity concrete" 
        else:
            assessment = "High porosity concrete"
        
        stats = {
            'global': global_porosity,
            'mean_local': mean_local,
            'median_local': median_local,
            'std_dev': std_dev,
            'coef_var': coef_var,
            'min_local': min_local,
            'max_local': max_local,
            'assessment': assessment
        }
        
        return local_porosity, grid_size, cell_width, cell_height, stats

    def show_porosity_analysis(self):
        """Display porosity analysis in a new window"""
        if self.pore_mask is None:
            self.analysis_status.config(text="No pore mask available. Process an image first.", fg="red")
            return
        
        # Calculate spatial porosity
        local_porosity, grid_size, cell_width, cell_height, stats = self.analyze_spatial_porosity(self.pore_mask)
        if local_porosity is None:
            self.analysis_status.config(text="No porosity data available for analysis", fg="red")
            return
        
        # Get units for display
        area_unit = self.get_area_unit()
        length_unit = self.get_length_unit()
        
        # Create a new window for the analysis
        porosity_win = tk.Toplevel(self.root)
        porosity_win.title("Porosity Analysis")
        porosity_win.geometry("1000x800")
        
        # Create notebook with tabs for different visualizations
        notebook = ttk.Notebook(porosity_win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Statistical Analysis - UPDATED WITH UNITS
        stats_tab = Frame(notebook)
        notebook.add(stats_tab, text="Statistics")
        
        # Create statistics display
        stats_frame = Frame(stats_tab)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title and columns
        tk.Label(stats_frame, text="Porosity Statistical Analysis", 
                font=("Arial", 14, "bold")).pack(anchor=tk.W, pady=(0,20))

        col1 = Frame(stats_frame)
        col1.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        
        col2 = Frame(stats_frame)
        col2.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        
        # Fill in statistics with units where appropriate
        tk.Label(col1, text="Global Statistics", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0,10))
        tk.Label(col1, text=f"Global Porosity: {stats['global']:.2f}%", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Mean Local Porosity: {stats['mean_local']:.2f}%", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Median Local Porosity: {stats['median_local']:.2f}%", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Minimum Local Porosity: {stats['min_local']:.2f}%", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Maximum Local Porosity: {stats['max_local']:.2f}%", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        
        # Add grid cell size information with units
        cell_area_display = self.area_px2_to_cm2(cell_width * cell_height)
        tk.Label(col1, text=f"\nGrid Analysis Info:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10,5))
        tk.Label(col1, text=f"Grid Size: {grid_size}×{grid_size} cells", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Cell Size: {self.px_to_cm(cell_width):.2f}×{self.px_to_cm(cell_height):.2f} {length_unit}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Cell Area: {cell_area_display:.2f} {area_unit}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        
        tk.Label(col2, text="Distribution Metrics", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0,10))
        tk.Label(col2, text=f"Standard Deviation: {stats['std_dev']:.2f}%", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col2, text=f"Coefficient of Variation: {stats['coef_var']:.2f}%", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col2, text=f"Porosity Assessment:", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col2, text=f"{stats['assessment']}", font=("Arial", 10, "bold"), fg="blue").pack(anchor=tk.W, pady=2)
        
        # Tab 2: Spatial Distribution - UPDATE WITH SEABORN STYLING
        spatial_tab = Frame(notebook)
        notebook.add(spatial_tab, text="Spatial Distribution")
        
        fig1 = Figure(figsize=(8, 6), dpi=100)
        ax1 = fig1.add_subplot(111)
        sns.set_style("dark")
        
        # Use a more vibrant colormap for the heatmap
        cax = ax1.imshow(local_porosity, cmap='viridis', interpolation='nearest')
        ax1.set_title('Spatial Porosity Distribution', fontsize=28)
        ax1.set_xlabel('X position', fontsize=24)
        ax1.set_ylabel('Y position', fontsize=24)
        ax1.tick_params(labelsize=20)
        cbar = fig1.colorbar(cax, label='Porosity %')
        cbar.set_label('Porosity %', fontsize=24)
        cbar.ax.tick_params(labelsize=20)
        
        canvas1 = FigureCanvasTkAgg(fig1, master=spatial_tab)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: 3D Porosity Map - UPDATE WITH SEABORN STYLING
        map3d_tab = Frame(notebook)
        notebook.add(map3d_tab, text="3D Porosity Map")
        
        fig3d = Figure(figsize=(8, 6), dpi=100)
        ax3d = fig3d.add_subplot(111, projection='3d')
        
        x = np.arange(0, grid_size)
        y = np.arange(0, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Use a cooler colormap for 3D visualization
        surf = ax3d.plot_surface(X, Y, local_porosity, cmap='plasma', 
                                linewidth=0, antialiased=True, alpha=0.8)
        ax3d.set_title('3D Porosity Distribution', fontsize=28)
        ax3d.set_xlabel('X Grid Position', fontsize=24)
        ax3d.set_ylabel('Y Grid Position', fontsize=24)
        ax3d.set_zlabel('Porosity %', fontsize=24)
        ax3d.tick_params(labelsize=20)
        
        # Improve 3D viewing angle
        ax3d.view_init(elev=30, azim=45)
        
        # Add grid to the plot
        ax3d.xaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.6)})
        ax3d.yaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.6)})
        ax3d.zaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.6)})
        
        fig3d.colorbar(surf, shrink=0.5, aspect=5, label='Porosity %').ax.tick_params(labelsize=20)
        
        canvas3d = FigureCanvasTkAgg(fig3d, master=map3d_tab)
        canvas3d.draw()
        canvas3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 4: Porosity Distribution Histogram - UPDATE WITH SEABORN STYLING
        hist_tab = Frame(notebook)
        notebook.add(hist_tab, text="Porosity Distribution")
        
        fig_hist = Figure(figsize=(8, 6), dpi=100)
        ax_hist = fig_hist.add_subplot(111)
        sns.set_style("whitegrid")
        
        # Flatten the local porosity array and filter out zeros
        flat_porosity = local_porosity.flatten()
        valid_porosity = flat_porosity[flat_porosity > 0]
        
        # Create histogram with Seaborn
        if len(valid_porosity) > 0:
            # Define bins based on the range of porosity values
            bin_min = max(0, np.floor(np.min(valid_porosity)))
            bin_max = np.ceil(np.max(valid_porosity))
            bins = np.linspace(bin_min, bin_max, 20)  # 20 bins
            
            # Use Seaborn's distplot with a cool colormap
            sns.histplot(valid_porosity, bins=bins, ax=ax_hist, 
                       color='skyblue', edgecolor='darkblue', 
                       kde=True, line_kws={'color': 'darkblue', 'lw': 2})
            
            ax_hist.axvline(stats['mean_local'], color='crimson', linestyle='--', lw=2,
                           label=f'Mean ({stats["mean_local"]:.2f}%)')
            ax_hist.axvline(stats['median_local'], color='forestgreen', linestyle=':', lw=2,
                           label=f'Median ({stats["median_local"]:.2f}%)')
            
            ax_hist.set_title('Porosity Distribution Histogram', fontsize=28)
            ax_hist.set_xlabel('Local Porosity %', fontsize=24)
            ax_hist.set_ylabel('Frequency', fontsize=24)
            ax_hist.tick_params(labelsize=20)
            ax_hist.legend(fontsize=20)
            
            # Add some grid styling
            ax_hist.grid(True, linestyle='--', alpha=0.7)
        else:
            ax_hist.text(0.5, 0.5, "No valid porosity data for histogram", 
                        ha='center', va='center', fontsize=24)
        
        canvas_hist = FigureCanvasTkAgg(fig_hist, master=hist_tab)
        canvas_hist.draw()
        canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update status
        self.analysis_status.config(text="Porosity analysis complete!", fg="green")

    def show_shape_analysis(self):
        """Display shape factor analysis with improved Seaborn styling"""
        if self.pore_mask is None:
            self.analysis_status.config(text="No pore mask available. Process an image first.", fg="red")
            return
    
        # Calculate contour data
        contours, _ = cv2.findContours(self.pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            self.analysis_status.config(text="No pores detected for shape analysis", fg="red")
            return
    
        # Calculate shape factors (calculations unchanged)
        areas = []
        perimeters = []
        shape_factors = []
        circularities = []
        diameters = []
    
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
    
            # Skip invalid contours
            if perimeter == 0 or area == 0:
                continue
    
            # Shape factor (4π*area/perimeter²)
            shape_factor = (4 * math.pi * area) / (perimeter * perimeter)
    
            # Add to lists if valid
            areas.append(area)
            perimeters.append(perimeter)
            shape_factors.append(shape_factor)
    
            # Circularity (same as shape factor but named differently)
            circularities.append(shape_factor)
    
            # Equivalent diameter
            diameter = 2 * math.sqrt(area / math.pi)
            diameters.append(diameter)
    
        # Convert to display units
        areas_disp = [self.area_px2_to_cm2(a) for a in areas]
        diameters_disp = [self.px_to_cm(d) for d in diameters]
        area_unit = self.get_area_unit()
        length_unit = self.get_length_unit()
    
        # Create a new window for the analysis
        shape_win = tk.Toplevel(self.root)
        shape_win.title("Shape Factor Analysis")
        shape_win.geometry("1000x800")
    
        # Create notebook with tabs for different visualizations
        notebook = ttk.Notebook(shape_win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
        # Tab 1: Shape Factor Distribution - UPDATED WITH SEABORN
        dist_tab = Frame(notebook)
        notebook.add(dist_tab, text="Shape Factor Distribution")
    
        fig_dist = Figure(figsize=(8, 5), dpi=100)
        ax_dist = fig_dist.add_subplot(111)
    
        # Apply Seaborn style
        sns.set_style("whitegrid")
    
        # Create histogram with Seaborn
        bins = np.linspace(0, 1, 20)
        sns.histplot(shape_factors, bins=bins, ax=ax_dist, 
                   color='#5ab4ac', edgecolor='#01665e', 
                   kde=True, line_kws={'color': '#01665e', 'linewidth': 2})
    
        ax_dist.axvline(np.mean(shape_factors), color='#d8b365', linestyle='--', linewidth=2.5,
                       label=f'Mean ({np.mean(shape_factors):.3f})')
        ax_dist.axvline(np.median(shape_factors), color='#5e3c99', linestyle=':', linewidth=2.5,
                       label=f'Median ({np.median(shape_factors):.3f})')
    
        ax_dist.set_title('Shape Factor Distribution', fontsize=28)
        ax_dist.set_xlabel('Shape Factor', fontsize=24)
        ax_dist.set_ylabel('Frequency', fontsize=24)
        ax_dist.tick_params(labelsize=20)
        ax_dist.legend(fontsize=20)
        ax_dist.grid(True, linestyle='--', alpha=0.7)
    
        canvas_dist = FigureCanvasTkAgg(fig_dist, master=dist_tab)
        canvas_dist.draw()
        canvas_dist.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
        # Tab 2: Shape Factor vs Area - UPDATED WITH SEABORN
        scatter_tab = Frame(notebook)
        notebook.add(scatter_tab, text="Shape Factor vs Area")
    
        fig_scatter = Figure(figsize=(8, 5), dpi=100)
        ax_scatter = fig_scatter.add_subplot(111)
    
        # Use Seaborn scatterplot with better color palette
        sns.scatterplot(x=areas_disp, y=shape_factors, ax=ax_scatter, 
                      alpha=0.7, color='#4575b4', edgecolor='#313695', s=50)
    
        ax_scatter.set_title('Shape Factor vs Pore Area', fontsize=28)
        ax_scatter.set_xlabel(f'Pore Area ({area_unit})', fontsize=24)
        ax_scatter.set_ylabel('Shape Factor', fontsize=24)
        ax_scatter.tick_params(labelsize=20)
    
        # Use log scale for area if range is large
        if max(areas_disp) / (min(areas_disp) + 1) > 100:
            ax_scatter.set_xscale('log')
    
        canvas_scatter = FigureCanvasTkAgg(fig_scatter, master=scatter_tab)
        canvas_scatter.draw()
        canvas_scatter.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
        # Tab 3: Shape Factor Box Plot - UPDATED WITH SEABORN
        box_tab = Frame(notebook)
        notebook.add(box_tab, text="Shape Factor Box Plot")
    
        fig_box = Figure(figsize=(8, 5), dpi=100)
        ax_box = fig_box.add_subplot(111)
    
        # Use Seaborn boxplot
        sns.boxplot(y=shape_factors, ax=ax_box, color='#a6bddb', 
                   width=0.3, linewidth=2.5, fliersize=5)
    
        ax_box.set_title('Shape Factor Box Plot', fontsize=28)
        ax_box.set_ylabel('Shape Factor', fontsize=24)
        ax_box.tick_params(labelsize=20)
        ax_box.set_xticks([])  # Remove x-axis ticks for cleaner look
    
        canvas_box = FigureCanvasTkAgg(fig_box, master=box_tab)
        canvas_box.draw()
        canvas_box.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
        # Tab 4: Statistics - UPDATED WITH UNITS
        stats_tab = Frame(notebook)
        notebook.add(stats_tab, text="Statistics")
    
        # Calculate statistics (unchanged)
        mean_sf = np.mean(shape_factors)
        median_sf = np.median(shape_factors)
        std_sf = np.std(shape_factors)
        min_sf = np.min(shape_factors)
        max_sf = np.max(shape_factors)
    
        # Circularity assessment
        if mean_sf > 0.9:
            circ_assessment = "Very circular pores (nearly perfect circles)"
        elif mean_sf > 0.8:
            circ_assessment = "Highly circular pores"
        elif mean_sf > 0.7:
            circ_assessment = "Moderately circular pores"
        elif mean_sf > 0.6:
            circ_assessment = "Somewhat irregular pores"
        elif mean_sf > 0.4:
            circ_assessment = "Irregular pores"
        else:
            circ_assessment = "Very irregular pores (far from circular)"
    
        # Create statistics frame with two columns
        stats_frame = Frame(stats_tab)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
        # Title
        tk.Label(stats_frame, text="Shape Factor Statistics", 
                font=("Arial", 14, "bold")).pack(anchor=tk.W, pady=(0,20))
    
        # Display statistics in two columns
        col1 = Frame(stats_frame)
        col1.pack(side=tk.LEFT, fill=tk.Y, expand=True)
    
        col2 = Frame(stats_frame)
        col2.pack(side=tk.RIGHT, fill=tk.Y, expand=True)
    
        # Column 1 - Basic stats
        tk.Label(col1, text="Numerical Statistics", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0,10))
        tk.Label(col1, text=f"Number of pores: {len(shape_factors)}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Mean shape factor: {mean_sf:.3f}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Median shape factor: {median_sf:.3f}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Standard deviation: {std_sf:.3f}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Minimum shape factor: {min_sf:.3f}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Maximum shape factor: {max_sf:.3f}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
    
        # Add area and diameter statistics with units
        tk.Label(col1, text=f"\nArea Statistics ({area_unit}):", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Mean area: {np.mean(areas_disp):.2f} {area_unit}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Mean diameter: {np.mean(diameters_disp):.2f} {length_unit}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
    
        # Column 2 - Assessment
        tk.Label(col2, text="Shape Assessment", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0,10))
        tk.Label(col2, text="Circularity Assessment:", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col2, text=circ_assessment, font=("Arial", 10, "bold"), fg="blue").pack(anchor=tk.W, pady=2)
    
        tk.Label(col2, text="\nShape Factor Interpretation:", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col2, text="1.0 = Perfect circle", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col2, text="<0.8 = Irregular shapes", font=("Arial", 10)).pack(anchor=tk.W, pady=2)

        # Update status
        self.analysis_status.config(text="Shape factor analysis complete!", fg="green")

    def show_size_analysis(self):
        """Display pore size distribution analysis in a new window"""
        if self.pore_mask is None:
            self.analysis_status.config(text="No pore mask available. Process an image first.", fg="red")
            return
        
        # Calculate contour data
        contours, _ = cv2.findContours(self.pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            self.analysis_status.config(text="No pores detected for size analysis", fg="red")
            return
        
        # Calculate areas and diameters
        areas = []
        diameters = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0:
                areas.append(area)
                # Calculate equivalent diameter (diameter of circle with same area)
                diameter = 2 * math.sqrt(area / math.pi)
                diameters.append(diameter)
        
        # Convert to display units
        areas_disp = [self.area_px2_to_cm2(a) for a in areas]
        diameters_disp = [self.px_to_cm(d) for d in diameters]
        area_unit = self.get_area_unit()
        length_unit = self.get_length_unit()
        
        # Create a new window for the analysis
        size_win = tk.Toplevel(self.root)
        size_win.title("Pore Size Analysis")
        size_win.geometry("1000x800")
        
        # Create notebook with tabs for different visualizations
        notebook = ttk.Notebook(size_win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Area Distribution
        area_tab = Frame(notebook)
        notebook.add(area_tab, text="Area Distribution")
        
        fig_area = Figure(figsize=(8, 5), dpi=100)
        ax_area = fig_area.add_subplot(111)
        
        # Determine appropriate bins based on data range
        if max(areas_disp) / min(areas_disp) > 1000:
            bins = np.logspace(np.log10(min(areas_disp)), np.log10(max(areas_disp)), 20)
            ax_area.set_xscale('log')
        else:
            bins = 20
        
        ax_area.hist(areas_disp, bins=bins, color='lightgreen', edgecolor='black')
        ax_area.axvline(np.mean(areas_disp), color='r', linestyle='--', 
                       label=f'Mean ({np.mean(areas_disp):.2f} {area_unit})')
        ax_area.axvline(np.median(areas_disp), color='g', linestyle=':', 
                       label=f'Median ({np.median(areas_disp):.2f} {area_unit})')
        
        ax_area.set_title('Pore Area Distribution', fontsize=28)
        ax_area.set_xlabel(f'Pore Area ({area_unit})', fontsize=24)
        ax_area.set_ylabel('Frequency', fontsize=24)
        ax_area.tick_params(labelsize=20)
        ax_area.legend(fontsize=20)
        
        canvas_area = FigureCanvasTkAgg(fig_area, master=area_tab)
        canvas_area.draw()
        canvas_area.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 2: Diameter Distribution
        diam_tab = Frame(notebook)
        notebook.add(diam_tab, text="Diameter Distribution")
        
        fig_diam = Figure(figsize=(8, 5), dpi=100)
        ax_diam = fig_diam.add_subplot(111)
        
        ax_diam.hist(diameters_disp, bins=20, color='lightblue', edgecolor='black')
        ax_diam.axvline(np.mean(diameters_disp), color='r', linestyle='--', 
                       label=f'Mean ({np.mean(diameters_disp):.2f} {length_unit})')
        ax_diam.axvline(np.median(diameters_disp), color='g', linestyle=':', 
                       label=f'Median ({np.median(diameters_disp):.2f} {length_unit})')
        
        ax_diam.set_title('Pore Diameter Distribution', fontsize=28)
        ax_diam.set_xlabel(f'Equivalent Diameter ({length_unit})', fontsize=24)
        ax_diam.set_ylabel('Frequency', fontsize=24)
        ax_diam.tick_params(labelsize=20)
        ax_diam.legend(fontsize=20)
        
        canvas_diam = FigureCanvasTkAgg(fig_diam, master=diam_tab)
        canvas_diam.draw()
        canvas_diam.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: Cumulative Distribution
        cumul_tab = Frame(notebook)
        notebook.add(cumul_tab, text="Cumulative Distribution")
        
        fig_cumul = Figure(figsize=(8, 5), dpi=100)
        ax_cumul = fig_cumul.add_subplot(111)
        
        # Sort areas for cumulative plot
        sorted_areas = np.sort(areas_disp)
        cumulative = np.arange(1, len(sorted_areas) + 1) / len(sorted_areas) * 100
        
        # Plot cumulative distribution
        ax_cumul.plot(sorted_areas, cumulative, '-o', markersize=3)
        ax_cumul.set_title('Cumulative Size Distribution', fontsize=28)
        ax_cumul.set_xlabel(f'Pore Area ({area_unit})', fontsize=24)
        ax_cumul.set_ylabel('Cumulative Percentage (%)', fontsize=24)
        ax_cumul.tick_params(labelsize=20)
        
        # Mark D10, D50, D90 values
        d10_idx = np.searchsorted(cumulative, 10)
        d50_idx = np.searchsorted(cumulative, 50)
        d90_idx = np.searchsorted(cumulative, 90)
        
        if d10_idx < len(sorted_areas):
            d10 = sorted_areas[d10_idx]
            ax_cumul.axhline(y=10, color='g', linestyle='--', alpha=0.5)
            ax_cumul.axvline(x=d10, color='g', linestyle='--', alpha=0.5)
            ax_cumul.plot(d10, 10, 'go')
            ax_cumul.text(d10, 5, f'D10: {d10:.2f} {area_unit}', color='g', fontsize=16)
        
        if d50_idx < len(sorted_areas):
            d50 = sorted_areas[d50_idx]
            ax_cumul.axhline(y=50, color='r', linestyle='--', alpha=0.5)
            ax_cumul.axvline(x=d50, color='r', linestyle='--', alpha=0.5)
            ax_cumul.plot(d50, 50, 'ro')
            ax_cumul.text(d50, 45, f'D50: {d50:.2f} {area_unit}', color='r', fontsize=16)
        
        if d90_idx < len(sorted_areas):
            d90 = sorted_areas[d90_idx]
            ax_cumul.axhline(y=90, color='b', linestyle='--', alpha=0.5)
            ax_cumul.axvline(x=d90, color='b', linestyle='--', alpha=0.5)
            ax_cumul.plot(d90, 90, 'bo')
            ax_cumul.text(d90, 85, f'D90: {d90:.2f} {area_unit}', color='b', fontsize=16)
        
        # Use log scale if range is large
        if max(areas_disp) / min(areas_disp) > 100:
            ax_cumul.set_xscale('log')
        
        canvas_cumul = FigureCanvasTkAgg(fig_cumul, master=cumul_tab)
        canvas_cumul.draw()
        canvas_cumul.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 4: Statistics
        stats_tab = Frame(notebook)
        notebook.add(stats_tab, text="Statistics")
        
        # Calculate statistics with display units
        mean_area = np.mean(areas_disp)
        median_area = np.median(areas_disp)
        std_area = np.std(areas_disp)
        min_area = np.min(areas_disp)
        max_area = np.max(areas_disp)
        
        mean_diam = np.mean(diameters_disp)
        median_diam = np.median(diameters_disp)
        std_diam = np.std(diameters_disp)
        min_diam = np.min(diameters_disp)
        max_diam = np.max(diameters_disp)
        
        # Skewness assessment
        skewness = (mean_area - median_area) / std_area if std_area > 0 else 0
        
        if skewness > 1:
            size_assessment = "Highly skewed toward smaller pores"
        elif skewness > 0.5:
            size_assessment = "Moderately skewed toward smaller pores"
        elif skewness > 0.1:
            size_assessment = "Slightly skewed toward smaller pores"
        elif skewness > -0.1:
            size_assessment = "Normally distributed pore sizes"
        elif skewness > -0.5:
            size_assessment = "Slightly skewed toward larger pores"
        elif skewness > -1:
            size_assessment = "Moderately skewed toward larger pores"
        else:
            size_assessment = "Highly skewed toward larger pores"
        
        # Create statistics frame
        stats_frame = Frame(stats_tab)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        tk.Label(stats_frame, text="Pore Size Statistics", 
                font=("Arial", 14, "bold")).pack(anchor=tk.W, pady=(0,20))
        
        # Display statistics in two columns
        col1 = Frame(stats_frame)
        col1.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        
        col2 = Frame(stats_frame)
        col2.pack(side=tk.RIGHT, fill=tk.Y, expand=True)
        
        # Column 1 - Area stats
        tk.Label(col1, text=f"Area Statistics ({area_unit})", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0,10))
        tk.Label(col1, text=f"Number of pores: {len(areas)}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Mean area: {mean_area:.2f} {area_unit}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Median area: {median_area:.2f} {area_unit}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Standard deviation: {std_area:.2f} {area_unit}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Minimum area: {min_area:.2f} {area_unit}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col1, text=f"Maximum area: {max_area:.2f} {area_unit}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)

        # Column 2 - Diameter and assessment
        tk.Label(col2, text=f"Diameter Statistics ({length_unit})", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0,10))
        tk.Label(col2, text=f"Mean diameter: {mean_diam:.2f} {length_unit}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col2, text=f"Median diameter: {median_diam:.2f} {length_unit}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col2, text=f"Minimum diameter: {min_diam:.2f} {length_unit}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col2, text=f"Maximum diameter: {max_diam:.2f} {length_unit}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)

        tk.Label(col2, text="\nSize Distribution Assessment:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10,2))
        tk.Label(col2, text=f"Skewness: {skewness:.3f}", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
        tk.Label(col2, text=size_assessment, font=("Arial", 10, "bold"), fg="blue").pack(anchor=tk.W, pady=2)

        # Update status
        self.analysis_status.config(text="Size analysis complete!", fg="green")
    
    def display_to_original_coords(self, event_x, event_y):
        """
        Convert display coordinates to original image coordinates consistently
        This helper function ensures consistent coordinate mapping across all interactions
        """
        # Safety check if no image is loaded
        if self.original_image is None:
            return None, None
            
        # Get dimensions of the label holding the image
        img_width = self.original_image_label.winfo_width()
        img_height = self.original_image_label.winfo_height()
        
        # Get dimensions of the currently displayed image
        displayed_img = self.original_image_label.image
        if displayed_img is None:
            return None, None
            
        disp_width = displayed_img.width()
        disp_height = displayed_img.height()
        
        # Calculate offset due to centering of image in label
        x_offset = (img_width - disp_width) / 2
        y_offset = (img_height - disp_height) / 2
        
        # Adjust event coordinates for the offset
        x = event_x - x_offset
        y = event_y - y_offset
        
        # Check if click is within displayed image bounds
        if x < 0 or x >= disp_width or y < 0 or y >= disp_height:
            return None, None
        
        # Scale to original image dimensions
        orig_height, orig_width = self.original_image.shape[:2]
        x_scale = orig_width / disp_width
        y_scale = orig_height / disp_height
        
        x_orig = int(x * x_scale)
        y_orig = int(y * y_scale)
        
        # Ensure coordinates are within image bounds
        x_orig = max(0, min(x_orig, orig_width-1))
        y_orig = max(0, min(y_orig, orig_height-1))
        
        return x_orig, y_orig
 
    def reset_all(self):
        """Reset the entire application to initial state"""
        # Ask for confirmation
        if not messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all settings and data?"):
            return
            
        # Reset image data
        self.original_image = None
        self.current_image = None
        self.pore_mask = None
        self.pore_percentage = 0
        self.image_filename = None
        
        # Reset samples
        self.reset_samples()
        
        # Reset ROI
        self.clear_roi()
        
        # Reset scale
        self.clear_scale()
        
        # Reset parameters to defaults
        self.block_scale.set(11)
        self.c_scale.set(2)
        self.blur_scale.set(5)
        self.morph_scale.set(1)
        self.min_size_scale.set(30)
        self.max_size_scale.set(100000)
        self.similarity_scale.set(20)
        self.concrete_similarity_scale.set(20)
        self.edge_low_threshold.set(50)
        self.edge_high_threshold.set(150)
        self.edge_combine_mode.set(0)
        self.use_edge_detection.set(0)
        self.show_contours.set(1)
        self.show_stats.set(1)
        
        # Clear labels
        self.pore_percentage_label.config(text="Porosity: 0.00%")
        self.pore_count_label.config(text="Pore Count: 0")
        self.avg_size_label.config(text="Avg Pore Size: 0.00 px²")
        self.analyzed_area_label.config(text="Analyzed Area: Full Image")
        self.analysis_status.config(text="Select an analysis type above", fg="blue")
        
        # Clear displays
        self.original_image_label.config(image='')
        self.processed_image_label.config(image='')
        self.mask_image_label.config(image='')
        
        # Clear histogram
        for widget in self.histogram_frame.winfo_children():
            widget.destroy()
            
        # Reset notebook to first tab
        self.notebook.select(0)
        
        # Update title
        self.root.title("Concrete Pore Analyzer")
        
        messagebox.showinfo("Reset Complete", "Application has been reset to default state.")

    def start_scale_selection(self):
        """Start scale selection mode"""
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        self.scale_mode = True
        self.root.title("Concrete Pore Analyzer - DRAW SCALE LINE (Right-click and drag on image)")

    def clear_scale(self):
        """Clear the selected scale"""
        self.scale_line = None
        self.scale_length_mm = None
        self.mm_per_px = None
        self.scale_status_label.config(text="Scale: Not Set", fg="red")
        self.update_image()

    def on_scale_mouse_down(self, event):
        """Handle right mouse button down for scale selection"""
        if not self.scale_mode or self.original_image is None:
            return
        x_orig, y_orig = self.display_to_original_coords(event.x, event.y)
        if x_orig is None or y_orig is None:
            return
        self.scale_line = [(x_orig, y_orig), None]

    def on_scale_mouse_drag(self, event):
        """Handle right mouse drag for scale selection"""
        if not self.scale_mode or self.scale_line is None or self.original_image is None:
            return
        x_orig, y_orig = self.display_to_original_coords(event.x, event.y)
        if x_orig is None or y_orig is None:
            return
        self.scale_line[1] = (x_orig, y_orig)
        
        # Show live preview of scale line
        display_original = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB).copy()
        if self.scale_line[0] and self.scale_line[1]:
            cv2.line(display_original, self.scale_line[0], self.scale_line[1], (255, 0, 255), 6)
        
        # Update display
        display_pil = Image.fromarray(display_original)
        display_pil.thumbnail((400, 400), Image.LANCZOS)
        display_photo = ImageTk.PhotoImage(display_pil)
        self.original_image_label.config(image=display_photo)
        self.original_image_label.image = display_photo

    def on_scale_mouse_up(self, event):
        """Handle right mouse button release for scale selection"""
        if not self.scale_mode or self.scale_line is None or self.original_image is None:
            return
        x_orig, y_orig = self.display_to_original_coords(event.x, event.y)
        if x_orig is None or y_orig is None:
            return
        self.scale_line[1] = (x_orig, y_orig)
        
        # Calculate pixel distance
        (x1, y1), (x2, y2) = self.scale_line
        px_dist = math.hypot(x2 - x1, y2 - y1)
        
        if px_dist < 5:
            self.scale_status_label.config(text="Scale: Line too short, try again.", fg="red")
            self.scale_line = None
            self.scale_mode = False
            self.root.title("Concrete Pore Analyzer")
            self.update_image()
            return
        
        # Prompt user for real-world length
        length_mm = self.prompt_scale_length()
        if length_mm is not None and length_mm > 0:
            self.scale_length_mm = length_mm
            self.mm_per_px = length_mm / px_dist
            self.scale_status_label.config(
                text=f"Scale: {length_mm:.2f} mm = {px_dist:.1f} px ({self.mm_per_px:.4f} mm/px)", fg="green")
        else:
            self.scale_status_label.config(text="Scale: Not Set", fg="red")
            self.scale_line = None
        
        self.scale_mode = False
        self.root.title("Concrete Pore Analyzer")
        self.update_image()

    def prompt_scale_length(self):
        """Prompt the user to enter the real-world length of the scale line in mm"""
        import tkinter.simpledialog
        length_mm = tkinter.simpledialog.askfloat(
            "Enter Scale Length", 
            "Enter the real-world length of the drawn line (in mm):",
            minvalue=0.01, 
            parent=self.root)
        return length_mm

    # Helper functions for unit conversion
    def px_to_cm(self, value):
        """Convert pixels to mm if scale is set"""
        if self.mm_per_px is not None:
            return value * self.mm_per_px
        return value

    def area_px2_to_cm2(self, value):
        """Convert px² to mm² if scale is set"""
        if self.mm_per_px is not None:
            return value * (self.mm_per_px ** 2)
        return value

    def get_area_unit(self):
        """Get the appropriate area unit"""
        return "mm²" if self.mm_per_px is not None else "px²"

    def get_length_unit(self):
        """Get the appropriate length unit"""
        return "mm" if self.mm_per_px is not None else "px"

    def reset_detection_method(self):
        """Reset to adaptive thresholding method"""
        self.reset_samples()
        self.use_edge_detection.set(0)
        self.update_image()

    def finish_analysis(self):
        """Save all analysis results to a folder named after the image"""
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        if self.pore_mask is None:
            messagebox.showwarning("No Analysis", "Please process the image first.")
            return
        
        # Get the folder to save results
        folder_path = filedialog.askdirectory(title="Select folder to save analysis results")
        if not folder_path:
            return
        
        # Create analysis folder using image filename
        import os
        from datetime import datetime
        
        # Use image filename if available, otherwise use timestamp
        if self.image_filename:
            folder_name = self.image_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"analysis_{timestamp}"
        
        analysis_folder = os.path.join(folder_path, folder_name)
        
        try:
            os.makedirs(analysis_folder, exist_ok=True)
            
            # Save images
            self.save_analysis_images(analysis_folder)
            
            # Generate and save analysis figures
            self.save_analysis_figures(analysis_folder)
            
            # Save comprehensive parameters and statistics
            self.save_comprehensive_analysis(analysis_folder)
            
            messagebox.showinfo("Analysis Complete", 
                              f"All analysis results saved to:\n{analysis_folder}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save analysis: {str(e)}")

    def save_analysis_images(self, folder_path):
        """Save original image, detection result, and mask"""
        import os
        
        # Save original image
        original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(original_rgb)
        original_pil.save(os.path.join(folder_path, "01_original_image.png"))
        
        # Save pore detection result
        processed_image = original_rgb.copy()
        
        # Create colored mask for pore visualization
        colored_mask = np.zeros_like(processed_image)
        colored_mask[self.pore_mask > 0] = [255, 0, 0]  # Red for pores
        
        # Add contours if enabled
        if self.show_contours.get():
            contours, _ = cv2.findContours(self.pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(processed_image, contours, -1, (0, 255, 0), 2)
        
        # Blend mask with original
        alpha = 0.5
        blended = cv2.addWeighted(processed_image, 1-alpha, colored_mask, alpha, 0)
        
        # Add ROI rectangle if present
        if self.roi_rectangle is not None:
            x1, y1, x2, y2 = self.roi_rectangle
            cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 255), 6)
        
        # Add scale line if present
        if self.scale_line and self.scale_line[0] and self.scale_line[1]:
            cv2.line(blended, self.scale_line[0], self.scale_line[1], (255, 0, 255), 6)  # Bright magenta line, thicker
        
        detection_pil = Image.fromarray(blended)
        detection_pil.save(os.path.join(folder_path, "02_pore_detection_result.png"))
        
        # Save pure pore mask
        mask_display = np.zeros((self.pore_mask.shape[0], self.pore_mask.shape[1], 3), dtype=np.uint8)
        mask_display[self.pore_mask > 0] = [255, 255, 255]  # White for pores
        mask_pil = Image.fromarray(mask_display)
        mask_pil.save(os.path.join(folder_path, "03_pore_mask.png"))

    def save_analysis_figures(self, folder_path):
        """Generate and save all analysis figures"""
        import os
        
        # Only proceed if we have pore data
        if self.pore_mask is None:
            return
        
        contours, _ = cv2.findContours(self.pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return
        
        # Save Shape Analysis Figures
        self.save_shape_analysis_figures(folder_path, contours)
        
        # Save Size Analysis Figures
        self.save_size_analysis_figures(folder_path, contours)
        
        # Save Porosity Analysis Figures
        self.save_porosity_analysis_figures(folder_path)

    def save_shape_analysis_figures(self, folder_path, contours):
        """Generate and save shape analysis figures"""
        import os
        
        # Calculate shape factors
        areas = []
        shape_factors = []
        diameters = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0 or area == 0:
                continue
            
            shape_factor = (4 * math.pi * area) / (perimeter * perimeter)
            areas.append(area)
            shape_factors.append(shape_factor)
            
            diameter = 2 * math.sqrt(area / math.pi)
            diameters.append(diameter)
        
        if not shape_factors:
            return
        
        # Convert to display units
        areas_disp = [self.area_px2_to_cm2(a) for a in areas]
        area_unit = self.get_area_unit()
        
        # Shape Factor Distribution
        fig = Figure(figsize=(10, 6), dpi=150)
        ax = fig.add_subplot(111)
        
        bins = np.linspace(0, 1, 20)
        sns.histplot(shape_factors, bins=bins, ax=ax, color='#5ab4ac', 
                    edgecolor='#01665e', kde=True, line_kws={'color': '#01665e', 'linewidth': 2})
        
        ax.axvline(np.mean(shape_factors), color='#d8b365', linestyle='--', linewidth=2.5,
                   label=f'Mean ({np.mean(shape_factors):.3f})')
        ax.axvline(np.median(shape_factors), color='#5e3c99', linestyle=':', linewidth=2.5,
                   label=f'Median ({np.median(shape_factors):.3f})')
        
        ax.set_title('Shape Factor Distribution', fontsize=28)
        ax.set_xlabel('Shape Factor', fontsize=24)
        ax.set_ylabel('Frequency', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        fig.savefig(os.path.join(folder_path, "04_shape_factor_distribution.png"), dpi=150, bbox_inches='tight')
        
        # Shape Factor vs Area
        fig2 = Figure(figsize=(10, 6), dpi=150)
        ax2 = fig2.add_subplot(111)
        
        sns.scatterplot(x=areas_disp, y=shape_factors, ax=ax2, 
                       alpha=0.7, color='#4575b4', edgecolor='#313695', s=50)
        
        ax2.set_title('Shape Factor vs Pore Area', fontsize=28)
        ax2.set_xlabel(f'Pore Area ({area_unit})', fontsize=24)
        ax2.set_ylabel('Shape Factor', fontsize=24)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        
        if max(areas_disp) / (min(areas_disp) + 1) > 100:
            ax2.set_xscale('log')
        
        fig2.tight_layout()
        fig2.savefig(os.path.join(folder_path, "05_shape_factor_vs_area.png"), dpi=150, bbox_inches='tight')

    def save_size_analysis_figures(self, folder_path, contours):
        """Generate and save size analysis figures"""
        import os
        
        # Calculate areas and diameters
        areas = []
        diameters = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0:
                areas.append(area)
                diameter = 2 * math.sqrt(area / math.pi)
                diameters.append(diameter)
        
        if not areas:
            return
        
        # Convert to display units
        areas_disp = [self.area_px2_to_cm2(a) for a in areas]
        diameters_disp = [self.px_to_cm(d) for d in diameters]
        area_unit = self.get_area_unit()
        length_unit = self.get_length_unit()
        
        # Area Distribution
        fig1 = Figure(figsize=(10, 6), dpi=150)
        ax1 = fig1.add_subplot(111)
        
        if max(areas_disp) / min(areas_disp) > 1000:
            bins = np.logspace(np.log10(min(areas_disp)), np.log10(max(areas_disp)), 20)
            ax1.set_xscale('log')
        else:
            bins = 20
        
        ax1.hist(areas_disp, bins=bins, color='lightgreen', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(areas_disp), color='r', linestyle='--', 
                   label=f'Mean ({np.mean(areas_disp):.2f} {area_unit})')
        ax1.axvline(np.median(areas_disp), color='g', linestyle=':', 
                   label=f'Median ({np.median(areas_disp):.2f} {area_unit})')
        
        ax1.set_title('Pore Area Distribution', fontsize=28)
        ax1.set_xlabel(f'Pore Area ({area_unit})', fontsize=24)
        ax1.set_ylabel('Frequency', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.legend(fontsize=20)
        ax1.grid(True, alpha=0.3)
        
        fig1.tight_layout()
        fig1.savefig(os.path.join(folder_path, "06_area_distribution.png"), dpi=150, bbox_inches='tight')
        
        # Diameter Distribution
        fig2 = Figure(figsize=(10, 6), dpi=150)
        ax2 = fig2.add_subplot(111)
        
        ax2.hist(diameters_disp, bins=20, color='lightblue', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(diameters_disp), color='r', linestyle='--', 
                   label=f'Mean ({np.mean(diameters_disp):.2f} {length_unit})')
        ax2.axvline(np.median(diameters_disp), color='g', linestyle=':', 
                   label=f'Median ({np.median(diameters_disp):.2f} {length_unit})')
        
        ax2.set_title('Pore Diameter Distribution', fontsize=28)
        ax2.set_xlabel(f'Equivalent Diameter ({length_unit})', fontsize=24)
        ax2.set_ylabel('Frequency', fontsize=24)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.legend(fontsize=20)
        ax2.grid(True, alpha=0.3)
        
        fig2.tight_layout()
        fig2.savefig(os.path.join(folder_path, "07_diameter_distribution.png"), dpi=150, bbox_inches='tight')
        
        # Cumulative Distribution
        fig3 = Figure(figsize=(10, 6), dpi=150)
        ax3 = fig3.add_subplot(111)
        
        sorted_areas = np.sort(areas_disp)
        cumulative = np.arange(1, len(sorted_areas) + 1) / len(sorted_areas) * 100
        
        ax3.plot(sorted_areas, cumulative, '-o', markersize=3, color='blue')
        ax3.set_title('Cumulative Size Distribution', fontsize=28)
        ax3.set_xlabel(f'Pore Area ({area_unit})', fontsize=24)
        ax3.set_ylabel('Cumulative Percentage (%)', fontsize=24)
        ax3.tick_params(axis='both', which='major', labelsize=20)
        ax3.grid(True, alpha=0.3)
        
        if max(areas_disp) / min(areas_disp) > 100:
            ax3.set_xscale('log')
        
        fig3.tight_layout()
        fig3.savefig(os.path.join(folder_path, "08_cumulative_distribution.png"), dpi=150, bbox_inches='tight')

    def save_porosity_analysis_figures(self, folder_path):
        """Generate and save porosity analysis figures"""
        import os
        
        # Get spatial porosity data
        local_porosity, grid_size, cell_width, cell_height, stats = self.analyze_spatial_porosity(self.pore_mask)
        if local_porosity is None:
            return
        
        # Spatial Distribution Heatmap
        fig1 = Figure(figsize=(10, 8), dpi=150)
        ax1 = fig1.add_subplot(111)
        
        cax = ax1.imshow(local_porosity, cmap='viridis', interpolation='nearest')
        ax1.set_title('Spatial Porosity Distribution', fontsize=28)
        ax1.set_xlabel('X position', fontsize=24)
        ax1.set_ylabel('Y position', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        cbar = fig1.colorbar(cax, label='Porosity %')
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Porosity %', fontsize=20)
        
        fig1.tight_layout()
        fig1.savefig(os.path.join(folder_path, "09_spatial_porosity_distribution.png"), dpi=150, bbox_inches='tight')
        
        # 3D Porosity Map
        fig2 = Figure(figsize=(12, 8), dpi=150)
        ax2 = fig2.add_subplot(111, projection='3d')
        
        x = np.arange(0, grid_size)
        y = np.arange(0, grid_size)
        X, Y = np.meshgrid(x, y)
        
        surf = ax2.plot_surface(X, Y, local_porosity, cmap='plasma', 
                               linewidth=0, antialiased=True, alpha=0.8)
        ax2.set_title('3D Porosity Distribution', fontsize=28)
        ax2.set_xlabel('X Grid Position', fontsize=24)
        ax2.set_ylabel('Y Grid Position', fontsize=24)
        ax2.set_zlabel('Porosity %', fontsize=24)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.view_init(elev=30, azim=45)
        
        cbar = fig2.colorbar(surf, shrink=0.5, aspect=5, label='Porosity %')
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Porosity %', fontsize=20)
        fig2.tight_layout()
        fig2.savefig(os.path.join(folder_path, "10_3d_porosity_map.png"), dpi=150, bbox_inches='tight')
        
        # Porosity Distribution Histogram
        fig3 = Figure(figsize=(10, 6), dpi=150)
        ax3 = fig3.add_subplot(111)
        
        flat_porosity = local_porosity.flatten()
        valid_porosity = flat_porosity[flat_porosity > 0]
        
        if len(valid_porosity) > 0:
            bin_min = max(0, np.floor(np.min(valid_porosity)))
            bin_max = np.ceil(np.max(valid_porosity))
            bins = np.linspace(bin_min, bin_max, 20)
            
            sns.histplot(valid_porosity, bins=bins, ax=ax3, 
                        color='skyblue', edgecolor='darkblue', 
                        kde=True, line_kws={'color': 'darkblue', 'lw': 2})
            
            ax3.axvline(stats['mean_local'], color='crimson', linestyle='--', lw=2,
                       label=f'Mean ({stats["mean_local"]:.2f}%)')
            ax3.axvline(stats['median_local'], color='forestgreen', linestyle=':', lw=2,
                       label=f'Median ({stats["median_local"]:.2f}%)')
            
            ax3.set_title('Porosity Distribution Histogram', fontsize=28)
            ax3.set_xlabel('Local Porosity %', fontsize=24)
            ax3.set_ylabel('Frequency', fontsize=24)
            ax3.tick_params(axis='both', which='major', labelsize=20)
            ax3.legend(fontsize=20)
            ax3.grid(True, linestyle='--', alpha=0.7)
        
        fig3.tight_layout()
        fig3.savefig(os.path.join(folder_path, "11_porosity_histogram.png"), dpi=150, bbox_inches='tight')

    def save_comprehensive_analysis(self, folder_path):
        """Save comprehensive analysis report with all parameters and statistics"""
        import os
        from datetime import datetime
        
        report_path = os.path.join(folder_path, "analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE CONCRETE PORE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Basic Results
            f.write("BASIC ANALYSIS RESULTS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Global Porosity: {self.pore_percentage:.2f}%\n")
            
            # Get pore count and average size
            if self.pore_mask is not None:
                contours, _ = cv2.findContours(self.pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pore_count = len(contours)
                areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0]
                avg_size = np.mean(areas) if areas else 0
                
                area_unit = self.get_area_unit()
                f.write(f"Pore Count: {pore_count}\n")
                f.write(f"Average Pore Size: {self.area_px2_to_cm2(avg_size):.2f} {area_unit}\n")
                
                # Scale Information
                f.write(f"\nSCALE INFORMATION\n")
                f.write("-" * 17 + "\n")
                if self.mm_per_px is not None:
                    px_dist = math.hypot(self.scale_line[1][0]-self.scale_line[0][0], 
                                       self.scale_line[1][1]-self.scale_line[0][1])
                    f.write(f"Scale Set: {self.scale_length_mm:.2f} mm = {px_dist:.1f} px\n")
                    f.write(f"Scale Factor: {self.mm_per_px:.6f} mm/px\n")
                    f.write(f"All measurements in: mm/mm²\n")
                else:
                    f.write(f"Scale: Not set (measurements in px/px²)\n")
                
                # ROI Information
                f.write(f"\nANALYSIS REGION\n")
                f.write("-" * 15 + "\n")
                if self.analyze_roi_only.get() and self.roi_mask is not None:
                    if self.roi_rectangle:
                        x1, y1, x2, y2 = self.roi_rectangle
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        f.write(f"ROI Selected: {width}×{height} = {self.area_px2_to_cm2(area):.2f} {area_unit}\n")
                    else:
                        f.write("ROI: Custom mask applied\n")
                else:
                    height, width = self.original_image.shape[:2]
                    total_area = height * width
                    f.write(f"Full Image: {width}×{height} = {self.area_px2_to_cm2(total_area):.2f} {area_unit}\n")
                
                # Shape Analysis Statistics
                if len(areas) > 0:
                    shape_factors = []
                    diameters = []
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0 and area > 0:
                            shape_factor = (4 * math.pi * area) / (perimeter * perimeter)
                            shape_factors.append(shape_factor)
                            diameter = 2 * math.sqrt(area / math.pi)
                            diameters.append(diameter)
                    
                    if shape_factors:
                        f.write(f"\nSHAPE ANALYSIS STATISTICS\n")
                        f.write("-" * 25 + "\n")
                        f.write(f"Mean Shape Factor: {np.mean(shape_factors):.3f}\n")
                        f.write(f"Median Shape Factor: {np.median(shape_factors):.3f}\n")
                        f.write(f"Shape Factor Std Dev: {np.std(shape_factors):.3f}\n")
                        f.write(f"Min Shape Factor: {np.min(shape_factors):.3f}\n")
                        f.write(f"Max Shape Factor: {np.max(shape_factors):.3f}\n")
                        
                        mean_sf = np.mean(shape_factors)
                        if mean_sf > 0.9:
                            assessment = "Very circular pores"
                        elif mean_sf > 0.8:
                            assessment = "Highly circular pores"
                        elif mean_sf > 0.7:
                            assessment = "Moderately circular pores"
                        elif mean_sf > 0.6:
                            assessment = "Somewhat irregular pores"
                        elif mean_sf > 0.4:
                            assessment = "Irregular pores"
                        else:
                            assessment = "Very irregular pores"
                        f.write(f"Shape Assessment: {assessment}\n")
                    
                    # Size Analysis Statistics
                    length_unit = self.get_length_unit()
                    areas_disp = [self.area_px2_to_cm2(a) for a in areas]
                    diameters_disp = [self.px_to_cm(d) for d in diameters]
                    
                    f.write(f"\nSIZE ANALYSIS STATISTICS\n")
                    f.write("-" * 24 + "\n")
                    f.write(f"Mean Area: {np.mean(areas_disp):.2f} {area_unit}\n")
                    f.write(f"Median Area: {np.median(areas_disp):.2f} {area_unit}\n")
                    f.write(f"Min Area: {np.min(areas_disp):.2f} {area_unit}\n")
                    f.write(f"Max Area: {np.max(areas_disp):.2f} {area_unit}\n")
                    f.write(f"Area Std Dev: {np.std(areas_disp):.2f} {area_unit}\n")
                    
                    if diameters_disp:
                        f.write(f"Mean Diameter: {np.mean(diameters_disp):.2f} {length_unit}\n")
                        f.write(f"Median Diameter: {np.median(diameters_disp):.2f} {length_unit}\n")
                        f.write(f"Min Diameter: {np.min(diameters_disp):.2f} {length_unit}\n")
                        f.write(f"Max Diameter: {np.max(diameters_disp):.2f} {length_unit}\n")
                
                # Porosity Analysis Statistics
                local_porosity, grid_size, cell_width, cell_height, stats = self.analyze_spatial_porosity(self.pore_mask)
                if stats:
                    f.write(f"\nPOROSITY ANALYSIS STATISTICS\n")
                    f.write("-" * 28 + "\n")
                    f.write(f"Global Porosity: {stats['global']:.2f}%\n")
                    f.write(f"Mean Local Porosity: {stats['mean_local']:.2f}%\n")
                    f.write(f"Median Local Porosity: {stats['median_local']:.2f}%\n")
                    f.write(f"Porosity Std Dev: {stats['std_dev']:.2f}%\n")
                    f.write(f"Min Local Porosity: {stats['min_local']:.2f}%\n")
                    f.write(f"Max Local Porosity: {stats['max_local']:.2f}%\n")
                    f.write(f"Coefficient of Variation: {stats['coef_var']:.2f}%\n")
                    f.write(f"Assessment: {stats['assessment']}\n")
                    
                    cell_area_display = self.area_px2_to_cm2(cell_width * cell_height)
                    f.write(f"Grid Size: {grid_size}×{grid_size} cells\n")
                    f.write(f"Cell Size: {self.px_to_cm(cell_width):.2f}×{self.px_to_cm(cell_height):.2f} {length_unit}\n")
                    f.write(f"Cell Area: {cell_area_display:.2f} {area_unit}\n")
            
            # Processing Parameters
            f.write(f"\nPROCESSING PARAMETERS\n")
            f.write("-" * 21 + "\n")
            
            # Detection method
            if self.use_edge_detection.get():
                if self.sample_pore_color is not None:
                    detection_method = "Edge + Sample-based Detection"
                else:
                    detection_method = "Edge + Adaptive Thresholding"
                mode_names = ["Enhance", "Filter", "Standalone"]
                f.write(f"Detection Method: {detection_method}\n")
                f.write(f"Edge Detection Mode: {mode_names[self.edge_combine_mode.get()]}\n")
                f.write(f"Edge Low Threshold: {self.edge_low_threshold.get()}\n")
                f.write(f"Edge High Threshold: {self.edge_high_threshold.get()}\n")
            elif self.sample_pore_color is not None:
                detection_method = "Sample-based Color Detection"
                f.write(f"Detection Method: {detection_method}\n")
            else:
                detection_method = "Adaptive Thresholding"
                f.write(f"Detection Method: {detection_method}\n")
            
            # Sample information
            if self.sample_pore_color is not None:
                f.write(f"Sample Pore Color (BGR): {self.sample_pore_color}\n")
                f.write(f"Pore Color Similarity Threshold: {self.similarity_scale.get()}\n")
            if self.sample_concrete_color is not None:
                f.write(f"Sample Concrete Color (BGR): {self.sample_concrete_color}\n")
                f.write(f"Concrete Color Similarity Threshold: {self.concrete_similarity_scale.get()}\n")
            
            # Preprocessing parameters
            f.write(f"Blur Kernel Size: {self.blur_scale.get()}\n")
            f.write(f"Block Size: {self.block_scale.get()}\n")
            f.write(f"C Value: {self.c_scale.get()}\n")
            f.write(f"Morphological Operation Size: {self.morph_scale.get()}\n")
            f.write(f"Minimum Pore Size: {self.min_size_scale.get()} px²\n")
            f.write(f"Maximum Pore Size: {self.max_size_scale.get()} px²\n")
            
            # Display options
            f.write(f"Show Contours: {'Yes' if self.show_contours.get() else 'No'}\n")
            f.write(f"Show Statistics: {'Yes' if self.show_stats.get() else 'No'}\n")
            
            f.write(f"\nFILES SAVED\n")
            f.write("-" * 11 + "\n")
            f.write("01_original_image.png - Original input image\n")
            f.write("02_pore_detection_result.png - Processed image with pore detection\n")
            f.write("03_pore_mask.png - Binary pore mask\n")
            f.write("04_shape_factor_distribution.png - Shape factor histogram\n")
            f.write("05_shape_factor_vs_area.png - Shape factor vs area plot\n")
            f.write("06_area_distribution.png - Pore area distribution\n")
            f.write("07_diameter_distribution.png - Pore diameter distribution\n")
            f.write("08_cumulative_distribution.png - Cumulative size distribution\n")
            f.write("09_spatial_porosity_distribution.png - Spatial porosity heatmap\n")
            f.write("10_3d_porosity_map.png - 3D porosity visualization\n")
            f.write("11_porosity_histogram.png - Porosity distribution histogram\n")
            f.write("analysis_report.txt - This comprehensive report\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = ConcretePoreAnalyzer(root)
    root.mainloop()