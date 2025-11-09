# frozen_ui.py
import tkinter as tk
from tkinter import ttk, messagebox
import os
import threading
import subprocess
import sys  # ensure we use the same interpreter to run papermill
# Added optional Pillow import for icon scaling
try:
    from PIL import Image, ImageTk  # type: ignore
except ImportError:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore
from frozen_backend import EnvConfig, FrozenLakeCustom, value_iteration

CELL_SIZE = 40
ICON_MARGIN = 2  # margin inside cell for scaled icon
COLORS = {
    "start": "#87cefa",
    "goal": "#32cd32",
    "hole": "#444444",
    "free": "#e0f7fa",
    "agent": "#ffcc00"
}

class FrozenLakeUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Frozen Lake Configurable UI")
        self.cfg = EnvConfig()
        self.env = FrozenLakeCustom(self.cfg)
        self.policy = None
        self.auto_running = False
        self.policy_stale = True
        self.images = {}
        self.scaled_images = {}
        self.status_var = tk.StringVar(value="Ready")
        self._load_images()
        self._build_controls()
        self._build_canvas()
        self._update_status()
        self.draw()

    def _load_images(self):
        # Load optional icons if present
        base_dir = os.path.dirname(__file__)
        icon_dir = os.path.join(base_dir, "icons")
        def load(name, key):
            path = os.path.join(icon_dir, name)
            if os.path.exists(path):
                try:
                    # If Pillow available use it for scaling; else PhotoImage
                    if Image is not None:
                        img = Image.open(path)
                        self.images[key] = img
                    else:
                        self.images[key] = tk.PhotoImage(file=path)
                except Exception:
                    self.images[key] = None
            else:
                self.images[key] = None
        load("agent.png", "agent")
        load("start.png", "start")
        load("goal.png", "goal")
        load("hole.png", "hole")
        load("free.png", "free")
        self._prepare_scaled_images()

    def _prepare_scaled_images(self):
        # Create scaled PhotoImage objects sized to CELL_SIZE-2*ICON_MARGIN
        self.scaled_images = {}
        target_size = CELL_SIZE - 2 * ICON_MARGIN
        for key, img in self.images.items():
            if img is None:
                self.scaled_images[key] = None
                continue
            try:
                if Image is not None and isinstance(img, Image.Image):
                    # Backward-compatible resample enum (Pillow<9 uses Image.LANCZOS)
                    resample_enum = getattr(Image, "Resampling", Image)
                    resized = img.resize((target_size, target_size), resample=resample_enum.LANCZOS)
                    self.scaled_images[key] = ImageTk.PhotoImage(resized)
                else:
                    # Native PhotoImage: attempt subsample if larger
                    if hasattr(img, 'width') and img.width() != target_size:
                        # crude scaling: only shrink by integer factor
                        factor = max(1, img.width() // target_size)
                        try:
                            scaled = img.subsample(factor, factor)
                            self.scaled_images[key] = scaled
                        except Exception:
                            self.scaled_images[key] = img
                    else:
                        self.scaled_images[key] = img
            except Exception:
                self.scaled_images[key] = None

    def _build_controls(self):
        frm = ttk.Frame(self.root, padding=5)
        frm.grid(row=0, column=0, sticky="nw")
        # Entries
        self.rows_var = tk.IntVar(value=self.cfg.rows)
        self.cols_var = tk.IntVar(value=self.cfg.cols)
        self.gamma_var = tk.DoubleVar(value=self.cfg.gamma)
        self.theta_var = tk.DoubleVar(value=self.cfg.theta)
        self.slip_var = tk.BooleanVar(value=self.cfg.slippery)
        self.slip_prob_var = tk.DoubleVar(value=self.cfg.slip_prob)
        self.start_var = tk.StringVar(value=f"{self.cfg.start[0]},{self.cfg.start[1]}")
        self.goal_var = tk.StringVar(value=f"{self.cfg.goal[0]},{self.cfg.goal[1]}")
        holes_str = ",".join(f"{r}:{c}" for r, c in self.cfg.holes)
        self.holes_var = tk.StringVar(value=holes_str)

        # Apply slippery config instantly when toggled/changed
        def _slip_changed(name=None, index=None, mode=None):  # trace_add callback signature
            self._apply_slip_vars()
            return None
        self.slip_var.trace_add("write", _slip_changed)
        self.slip_prob_var.trace_add("write", _slip_changed)

        def add_row(label, var, width=8):
            r = ttk.Frame(frm)
            r.pack(anchor="w")
            ttk.Label(r, text=label, width=16).pack(side="left")
            if isinstance(var, (tk.BooleanVar,)):
                ttk.Checkbutton(r, variable=var).pack(side="left")
            else:
                ttk.Entry(r, textvariable=var, width=width).pack(side="left")

        add_row("Rows", self.rows_var)
        add_row("Cols", self.cols_var)
        add_row("Gamma", self.gamma_var)
        add_row("Theta", self.theta_var)
        add_row("Slippery", self.slip_var)
        add_row("Slip Prob", self.slip_prob_var)
        add_row("Start r,c", self.start_var)
        add_row("Goal r,c", self.goal_var)
        add_row("Holes r:c,...", self.holes_var)

        # Status label
        status_lbl = ttk.Label(frm, textvariable=self.status_var, foreground="#444")
        status_lbl.pack(anchor="w", pady=(4, 0))

        btns = ttk.Frame(frm)
        btns.pack(pady=4, anchor="w")
        ttk.Button(btns, text="Build Env", command=self.build_env).pack(side="left", padx=2)
        ttk.Button(btns, text="Compute Policy", command=self.compute_policy).pack(side="left", padx=2)
        ttk.Button(btns, text="Manual Reset", command=self.manual_reset).pack(side="left", padx=2)
        ttk.Button(btns, text="Step Policy", command=self.step_policy_once).pack(side="left", padx=2)
        ttk.Button(btns, text="Run Auto", command=self.run_auto).pack(side="left", padx=2)
        ttk.Button(btns, text="Stop Auto", command=self.stop_auto).pack(side="left", padx=2)
        ttk.Button(btns, text="Run Policy Notebook", command=self.run_notebook).pack(side="left", padx=2)

        # Click tool selection
        tool_frame = ttk.LabelFrame(frm, text="Click Tool", padding=4)
        tool_frame.pack(pady=6, anchor="w", fill="x")
        self.tool_var = tk.StringVar(value="hole")
        ttk.Radiobutton(tool_frame, text="Toggle Hole", value="hole", variable=self.tool_var).pack(side="left")
        ttk.Radiobutton(tool_frame, text="Set Start", value="start", variable=self.tool_var).pack(side="left")
        ttk.Radiobutton(tool_frame, text="Set Goal", value="goal", variable=self.tool_var).pack(side="left")

        direction = ttk.Frame(frm)
        direction.pack(pady=4)
        ttk.Button(direction, text="Left", command=lambda: self.manual_step(0)).grid(row=0, column=0)
        ttk.Button(direction, text="Down", command=lambda: self.manual_step(1)).grid(row=0, column=1)
        ttk.Button(direction, text="Right", command=lambda: self.manual_step(2)).grid(row=0, column=2)
        ttk.Button(direction, text="Up", command=lambda: self.manual_step(3)).grid(row=0, column=3)

    def _apply_slip_vars(self):
        # Update config live for slipperiness; clamp prob and refresh status
        try:
            self.cfg.slippery = bool(self.slip_var.get())
            p = float(self.slip_prob_var.get())
            if p < 0.0:
                p = 0.0
            if p > 1.0:
                p = 1.0
            # reflect clamped value back to UI if changed
            if p != self.slip_prob_var.get():
                self.slip_prob_var.set(p)
            self.cfg.slip_prob = p
            self.env.cfg = self.cfg
            self.policy_stale = True
            self._update_status()
        except Exception:
            pass

    def _build_canvas(self):
        self.canvas = tk.Canvas(
            self.root,
            width=self.cfg.cols * CELL_SIZE,
            height=self.cfg.rows * CELL_SIZE,
            bg="white",
            highlightthickness=0  # remove outer canvas border to avoid artifacts
        )
        self.canvas.grid(row=0, column=1, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.handle_click)

    def build_env(self):
        try:
            rows = self.rows_var.get()
            cols = self.cols_var.get()
            start_parts = [int(x) for x in self.start_var.get().split(",") if x.strip() != ""]
            goal_parts = [int(x) for x in self.goal_var.get().split(",") if x.strip() != ""]
            if len(start_parts) != 2 or len(goal_parts) != 2:
                raise ValueError("Start and Goal must be in format 'r,c' with two integers.")
            start: tuple[int, int] = (start_parts[0], start_parts[1])
            goal: tuple[int, int] = (goal_parts[0], goal_parts[1])  # fixed typo goalParts -> goal_parts
            if not (0 <= start[0] < rows and 0 <= start[1] < cols):
                raise ValueError("Start outside grid bounds.")
            if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
                raise ValueError("Goal outside grid bounds.")
            holes: list[tuple[int, int]] = []
            txt = self.holes_var.get().strip()
            if txt:
                for part in txt.split(","):
                    if not part:
                        continue
                    if ":" not in part:
                        raise ValueError(f"Hole entry '{part}' must be r:c")
                    r_str, c_str = part.split(":")
                    r, c = int(r_str), int(c_str)
                    if not (0 <= r < rows and 0 <= c < cols):
                        raise ValueError(f"Hole ({r},{c}) outside grid bounds.")
                    if (r, c) == start or (r, c) == goal:
                        # silently skip if overlaps start/goal
                        continue
                    holes.append((r, c))
            slippery = self.slip_var.get()
            slip_prob = self.slip_prob_var.get()
            # clamp
            if slip_prob < 0.0:
                slip_prob = 0.0
            if slip_prob > 1.0:
                slip_prob = 1.0
            self.slip_prob_var.set(slip_prob)
            gamma = self.gamma_var.get()
            theta = self.theta_var.get()
            self.cfg = EnvConfig(rows=rows, cols=cols, start=start, goal=goal,
                                 holes=holes, slippery=slippery, slip_prob=slip_prob,
                                 gamma=gamma, theta=theta)
            self.env = FrozenLakeCustom(self.cfg)
            self.policy = None
            self.policy_stale = True
            self._resize_canvas()
            self.env.reset()
            self._prepare_scaled_images()  # regenerate scaled images if dimension changed
            self._update_status()
            self.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _resize_canvas(self):
        self.canvas.config(width=self.cfg.cols * CELL_SIZE, height=self.cfg.rows * CELL_SIZE)

    def compute_policy(self):
        self.policy, _ = value_iteration(self.env)
        self.policy_stale = False
        messagebox.showinfo("Policy", "Optimal policy computed.")

    def manual_reset(self):
        self.stop_auto()
        self.env.reset()
        self.draw()

    def manual_step(self, action: int):
        if self.env.terminated:
            return
        self.env.step(action)
        self.draw()

    def step_policy_once(self):
        if self.policy is None or self.policy_stale:
            # Auto recompute to reflect current env (slippery etc.)
            self.compute_policy()
        if self.env.terminated:
            return
        s = self.env.state
        a = int(self.policy[s])
        self.env.step(a)
        self.draw()

    def run_auto(self):
        if self.policy is None or self.policy_stale:
            self.compute_policy()
        if self.auto_running:
            return
        self.auto_running = True
        self._auto_loop()

    def _auto_loop(self):
        if not self.auto_running or self.env.terminated:
            self.auto_running = False
            return
        s = self.env.state
        a = int(self.policy[s])
        self.env.step(a)
        self.draw()
        # Use direct reference to satisfy callable signature
        self.root.after(350, self._auto_loop)

    def stop_auto(self):
        self.auto_running = False

    def handle_click(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if not (0 <= row < self.cfg.rows and 0 <= col < self.cfg.cols):
            return
        mode = self.tool_var.get()
        if mode == "hole":
            # Toggle hole unless start/goal
            if (row, col) == self.cfg.start or (row, col) == self.cfg.goal:
                return
            holes = set(self.cfg.holes)
            if (row, col) in holes:
                holes.remove((row, col))
            else:
                holes.add((row, col))
            self.holes_var.set(",".join(f"{r}:{c}" for r, c in sorted(holes)))
            self.build_env()
            self.policy_stale = True
        elif mode == "start":
            if (row, col) == self.cfg.goal:
                messagebox.showwarning("Invalid", "Start cannot overlap Goal.")
                return
            # Remove hole if present
            holes = set(self.cfg.holes)
            holes.discard((row, col))
            self.holes_var.set(",".join(f"{r}:{c}" for r, c in sorted(holes)))
            self.start_var.set(f"{row},{col}")
            self.build_env()
            self.policy_stale = True
        elif mode == "goal":
            if (row, col) == self.cfg.start:
                messagebox.showwarning("Invalid", "Goal cannot overlap Start.")
                return
            holes = set(self.cfg.holes)
            holes.discard((row, col))
            self.holes_var.set(",".join(f"{r}:{c}" for r, c in sorted(holes)))
            self.goal_var.set(f"{row},{col}")
            self.build_env()
            self.policy_stale = True

    def draw(self):
        self.canvas.delete("all")
        for r in range(self.cfg.rows):
            for c in range(self.cfg.cols):
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                cell_type = "free"
                if (r, c) == self.cfg.start:
                    cell_type = "start"
                if (r, c) == self.cfg.goal:
                    cell_type = "goal"
                if (r, c) in self.cfg.holes:
                    cell_type = "hole"
                color = COLORS[cell_type]
                # Draw cell background without outline; borders are drawn by grid lines
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
                img = self.scaled_images.get(cell_type)
                if img is not None:
                    self.canvas.create_image(x0 + CELL_SIZE//2, y0 + CELL_SIZE//2, image=img)
        # Agent
        ar, ac = self.env.index_to_rc(self.env.state)
        agent_img = self.scaled_images.get("agent")
        if agent_img is not None:
            self.canvas.create_image(ac * CELL_SIZE + CELL_SIZE//2, ar * CELL_SIZE + CELL_SIZE//2, image=agent_img)
        else:
            ax0 = ac * CELL_SIZE + ICON_MARGIN
            ay0 = ar * CELL_SIZE + ICON_MARGIN
            ax1 = ax0 + CELL_SIZE - 2 * ICON_MARGIN
            ay1 = ay0 + CELL_SIZE - 2 * ICON_MARGIN
            self.canvas.create_oval(ax0, ay0, ax1, ay1, fill=COLORS["agent"], outline="black")
        # Draw crisp grid lines last so they sit on top
        for c in range(self.cfg.cols + 1):
            x = c * CELL_SIZE
            self.canvas.create_line(x, 0, x, self.cfg.rows * CELL_SIZE, fill="black", width=1)
        for r in range(self.cfg.rows + 1):
            y = r * CELL_SIZE
            self.canvas.create_line(0, y, self.cfg.cols * CELL_SIZE, y, fill="black", width=1)
        if self.env.terminated:
            ar, ac = self.env.index_to_rc(self.env.state)
            txt = "WIN" if (ar, ac) == self.cfg.goal else "FAIL"
            self.canvas.create_text(self.cfg.cols * CELL_SIZE // 2,
                                    self.cfg.rows * CELL_SIZE // 2,
                                    text=txt, font=("Arial", 24), fill="red")

    def run_notebook(self):
        # Run game_app/nootbok/Frosen_POLICY_ITERATION_to_call.ipynb with current UI parameters using papermill
        def target():
            try:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
                # Input notebook now lives inside game_app/nootbok
                nb_in = os.path.join(base_dir, "game_app", "nootbok", "Frosen_POLICY_ITERATION_to_call.ipynb")
                if not os.path.exists(nb_in):
                    messagebox.showerror("Notebook Missing", f"Input notebook not found:\n{nb_in}")
                    self.status_var.set("Notebook missing")
                    return
                out_dir = os.path.join(base_dir, "outputs")
                os.makedirs(out_dir, exist_ok=True)
                nb_out = os.path.join(out_dir, "Frosen_POLICY_ITERATION_out.ipynb")

                # Preflight: ensure papermill is available in current interpreter
                py = sys.executable or "python"
                check = subprocess.run([py, "-c", "import papermill, sys; sys.stdout.write(papermill.__version__)"] ,
                                       cwd=base_dir, capture_output=True, text=True)
                if check.returncode != 0:
                    install = messagebox.askyesno(
                        "Papermill Missing",
                        "Papermill is not installed in this Python environment.\n\n"
                        "Would you like to install it now?\n\n"
                        "This runs: pip install papermill",
                    )
                    if install:
                        self.status_var.set("Installing papermill…")
                        pip_proc = subprocess.run([py, "-m", "pip", "install", "papermill"],
                                                  cwd=base_dir, capture_output=True, text=True)
                        if pip_proc.returncode != 0:
                            messagebox.showerror("Install Failed", pip_proc.stderr or pip_proc.stdout)
                            self.status_var.set("Papermill installation failed")
                            return
                        else:
                            self.status_var.set("Papermill installed. Running notebook…")
                    else:
                        messagebox.showinfo(
                            "Papermill Required",
                            "Please install papermill and try again. Example:\n\n"
                            "pip install papermill\n\n"
                            f"Active Python: {py}")
                        return

                params = [
                    "-p", "rows", str(self.cfg.rows),
                    "-p", "cols", str(self.cfg.cols),
                    "-p", "start", f"({self.cfg.start[0]},{self.cfg.start[1]})",
                    "-p", "goal", f"({self.cfg.goal[0]},{self.cfg.goal[1]})",
                    "-p", "holes", "[" + ",".join(f"({r},{c})" for r, c in self.cfg.holes) + "]",
                    "-p", "slippery", str(bool(self.cfg.slippery)),
                    "-p", "slip_prob", str(float(self.cfg.slip_prob)),
                    "-p", "gamma", str(float(self.cfg.gamma)),
                    "-p", "theta", str(float(self.cfg.theta)),
                ]
                cmd = [py, "-m", "papermill", nb_in, nb_out] + params
                self.status_var.set("Running policy notebook…")
                proc = subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True)
                if proc.returncode == 0:
                    messagebox.showinfo(
                        "Notebook",
                        f"Notebook executed. Output saved to:\n{nb_out}\n\n"
                        f"Interpreter: {py}\n"
                        "Note: The notebook must accept papermill parameters with the same variable names.")
                    self.status_var.set("Notebook finished successfully")
                else:
                    err = proc.stderr.strip() or proc.stdout.strip() or "Unknown error"
                    messagebox.showerror("Notebook Error", err)
                    self.status_var.set("Notebook failed")
            except Exception as e:
                messagebox.showerror("Notebook Error", str(e))
                self.status_var.set("Notebook failed (exception)")
        threading.Thread(target=target, daemon=True).start()

    def _update_status(self):
        slip_txt = "ON" if self.cfg.slippery else "OFF"
        self.status_var.set(f"Slippery: {slip_txt}  p={self.cfg.slip_prob:.2f}  Size={self.cfg.rows}x{self.cfg.cols}")


def main():
    root = tk.Tk()
    FrozenLakeUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
