import tkinter as tk
from tkinter import ttk, messagebox
import os
import threading
import subprocess
import sys
try:
    from PIL import Image, ImageTk  # type: ignore
except ImportError:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore
from frozen_backend import EnvConfig, FrozenLakeCustom, value_iteration

CELL_SIZE = 40
ICON_MARGIN = 2
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
        base_dir = os.path.dirname(__file__)
        icon_dir = os.path.join(base_dir, "icons")
        def load(name, key):
            path = os.path.join(icon_dir, name)
            if os.path.exists(path):
                try:
                    if Image is not None:
                        img = Image.open(path)
                        self.images[key] = img
                    else:
                        self.images[key] = tk.PhotoImage(file=path)
                except Exception:
                    self.images[key] = None
            else:
                self.images[key] = None
        for name, key in [("agent.png","agent"),("start.png","start"),("goal.png","goal"),("hole.png","hole"),("free.png","free")]:
            load(name, key)
        self._prepare_scaled_images()

    def _prepare_scaled_images(self):
        self.scaled_images = {}
        target = CELL_SIZE - 2*ICON_MARGIN
        for key, img in self.images.items():
            if img is None:
                self.scaled_images[key] = None
                continue
            try:
                if Image is not None and isinstance(img, Image.Image):
                    resample_enum = getattr(Image, "Resampling", Image)
                    resized = img.resize((target, target), resample=resample_enum.LANCZOS)
                    self.scaled_images[key] = ImageTk.PhotoImage(resized)
                else:
                    self.scaled_images[key] = img
            except Exception:
                self.scaled_images[key] = None

    def _build_controls(self):
        frm = ttk.Frame(self.root, padding=5)
        frm.grid(row=0, column=0, sticky="nw")
        self.rows_var = tk.IntVar(value=self.cfg.rows)
        self.cols_var = tk.IntVar(value=self.cfg.cols)
        self.gamma_var = tk.DoubleVar(value=self.cfg.gamma)
        self.theta_var = tk.DoubleVar(value=self.cfg.theta)
        self.slip_var = tk.BooleanVar(value=self.cfg.slippery)
        self.slip_prob_var = tk.DoubleVar(value=self.cfg.slip_prob)
        self.start_var = tk.StringVar(value=f"{self.cfg.start[0]},{self.cfg.start[1]}")
        self.goal_var = tk.StringVar(value=f"{self.cfg.goal[0]},{self.cfg.goal[1]}")
        self.holes_var = tk.StringVar(value=",".join(f"{r}:{c}" for r,c in self.cfg.holes))

        def add_row(label, var, is_bool=False, width=8):
            rowf = ttk.Frame(frm); rowf.pack(anchor="w")
            ttk.Label(rowf, text=label, width=14).pack(side="left")
            if is_bool:
                ttk.Checkbutton(rowf, variable=var).pack(side="left")
            else:
                ttk.Entry(rowf, textvariable=var, width=width).pack(side="left")
        add_row("Rows", self.rows_var)
        add_row("Cols", self.cols_var)
        add_row("Gamma", self.gamma_var)
        add_row("Theta", self.theta_var)
        add_row("Slippery", self.slip_var, is_bool=True)
        add_row("Slip Prob", self.slip_prob_var)
        add_row("Start r,c", self.start_var)
        add_row("Goal r,c", self.goal_var)
        add_row("Holes", self.holes_var, width=20)

        # Use properly-signed callbacks for trace_add to satisfy type checkers
        self.slip_var.trace_add("write", self._on_slip_trace)
        self.slip_prob_var.trace_add("write", self._on_slip_trace)

        btns = ttk.Frame(frm); btns.pack(pady=4, anchor="w")
        for text, cmd in [
            ("Build Env", self.build_env),
            ("Compute Policy", self.compute_policy),
            ("Manual Reset", self.manual_reset),
            ("Step Policy", self.step_policy_once),
            ("Run Auto", self.run_auto),
            ("Stop Auto", self.stop_auto),
            ("Run Policy Notebook", self.run_notebook)
        ]:
            ttk.Button(btns, text=text, command=cmd).pack(side="left", padx=2)

        tool_frame = ttk.LabelFrame(frm, text="Click Tool", padding=4)
        tool_frame.pack(pady=6, anchor="w", fill="x")
        self.tool_var = tk.StringVar(value="hole")
        for lbl, val in [("Toggle Hole","hole"),("Set Start","start"),("Set Goal","goal")]:
            ttk.Radiobutton(tool_frame, text=lbl, value=val, variable=self.tool_var).pack(side="left")

        dirf = ttk.Frame(frm); dirf.pack(pady=4)
        for (lbl, a, col) in [("Left",0,0),("Down",1,1),("Right",2,2),("Up",3,3)]:
            ttk.Button(dirf, text=lbl, command=lambda act=a: self.manual_step(act)).grid(row=0, column=col)

        ttk.Label(frm, textvariable=self.status_var, foreground="#555").pack(anchor="w", pady=(4,0))

    def _apply_slip_vars(self):
        try:
            self.cfg.slippery = bool(self.slip_var.get())
            p = float(self.slip_prob_var.get())
            p = max(0.0, min(1.0, p))
            if p != self.slip_prob_var.get():
                self.slip_prob_var.set(p)
            self.cfg.slip_prob = p
            self.env.cfg = self.cfg
            self.policy_stale = True
            self._update_status()
        except Exception:
            pass

    def _build_canvas(self):
        self.canvas = tk.Canvas(self.root, width=self.cfg.cols*CELL_SIZE, height=self.cfg.rows*CELL_SIZE,
                                bg="white", highlightthickness=0)
        self.canvas.grid(row=0, column=1, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.handle_click)

    def build_env(self):
        try:
            rows = self.rows_var.get(); cols = self.cols_var.get()
            sp = [int(x) for x in self.start_var.get().split(',') if x.strip()]; gp = [int(x) for x in self.goal_var.get().split(',') if x.strip()]
            if len(sp)!=2 or len(gp)!=2: raise ValueError("Start/Goal must be 'r,c'.")
            start = (sp[0], sp[1]); goal = (gp[0], gp[1])
            if not (0<=start[0]<rows and 0<=start[1]<cols): raise ValueError("Start outside grid.")
            if not (0<=goal[0]<rows and 0<=goal[1]<cols): raise ValueError("Goal outside grid.")
            holes=[]; txt=self.holes_var.get().strip()
            if txt:
                for part in txt.split(','):
                    if not part: continue
                    if ':' not in part: raise ValueError(f"Hole '{part}' must be r:c")
                    r_str,c_str=part.split(':'); r=int(r_str); c=int(c_str)
                    if not (0<=r<rows and 0<=c<cols): raise ValueError(f"Hole ({r},{c}) outside grid.")
                    if (r,c) in [start, goal]: continue
                    holes.append((r,c))
            slippery = bool(self.slip_var.get()); slip_prob = float(self.slip_prob_var.get())
            slip_prob = max(0.0, min(1.0, slip_prob)); self.slip_prob_var.set(slip_prob)
            gamma = float(self.gamma_var.get()); theta = float(self.theta_var.get())
            self.cfg = EnvConfig(rows=rows, cols=cols, start=start, goal=goal, holes=holes,
                                 slippery=slippery, slip_prob=slip_prob, gamma=gamma, theta=theta)
            self.env = FrozenLakeCustom(self.cfg)
            self.policy = None; self.policy_stale = True
            self.canvas.config(width=cols*CELL_SIZE, height=rows*CELL_SIZE)
            self.env.reset(); self._prepare_scaled_images(); self._update_status(); self.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def compute_policy(self):
        self.policy, _ = value_iteration(self.env)
        self.policy_stale = False
        messagebox.showinfo("Policy", "Optimal policy computed.")

    def manual_reset(self):
        self.stop_auto(); self.env.reset(); self.draw()

    def manual_step(self, action: int):
        if self.env.terminated: return
        self.env.step(action); self.draw()

    def step_policy_once(self):
        if self.policy is None or self.policy_stale: self.compute_policy()
        if self.env.terminated: return
        s = self.env.state; a = int(self.policy[s]); self.env.step(a); self.draw()

    def run_auto(self):
        if self.policy is None or self.policy_stale: self.compute_policy()
        if self.auto_running: return
        self.auto_running = True; self._auto_loop()

    def _auto_loop(self):
        if not self.auto_running or self.env.terminated:
            self.auto_running = False
            return
        s = self.env.state
        a = int(self.policy[s])
        self.env.step(a)
        self.draw()
        self.root.after(300, lambda: self._after_auto_noargs())

    def _on_slip_trace(self, varname: str, index: str, mode: str):
        # trace callbacks receive (varname, index, mode)
        try:
            self._apply_slip_vars()
        except Exception:
            pass
        return None

    def _after_auto(self, *args):
        # wrapper for Tk.after which may pass extra args
        self._auto_loop()

    def _after_auto_noargs(self):
        # exact no-arg wrapper preferred by some static analyzers
        self._after_auto()

    def stop_auto(self):
        self.auto_running = False

    def handle_click(self, event):
        col = event.x // CELL_SIZE; row = event.y // CELL_SIZE
        if not (0<=row<self.cfg.rows and 0<=col<self.cfg.cols): return
        mode = self.tool_var.get()
        if mode == "hole":
            holes = set(self.cfg.holes)
            if (row,col) in [self.cfg.start, self.cfg.goal]: return
            if (row,col) in holes: holes.remove((row,col))
            else: holes.add((row,col))
            self.holes_var.set(",".join(f"{r}:{c}" for r,c in sorted(holes)))
            self.build_env(); self.policy_stale = True
        elif mode == "start":
            if (row,col) == self.cfg.goal: return
            holes = set(self.cfg.holes); holes.discard((row,col))
            self.holes_var.set(",".join(f"{r}:{c}" for r,c in sorted(holes)))
            self.start_var.set(f"{row},{col}"); self.build_env(); self.policy_stale = True
        elif mode == "goal":
            if (row,col) == self.cfg.start: return
            holes = set(self.cfg.holes); holes.discard((row,col))
            self.holes_var.set(",".join(f"{r}:{c}" for r,c in sorted(holes)))
            self.goal_var.set(f"{row},{col}"); self.build_env(); self.policy_stale = True

    def draw(self):
        self.canvas.delete("all")
        for r in range(self.cfg.rows):
            for c in range(self.cfg.cols):
                x0=c*CELL_SIZE; y0=r*CELL_SIZE; x1=x0+CELL_SIZE; y1=y0+CELL_SIZE
                cell="free"
                if (r,c)==self.cfg.start: cell="start"
                if (r,c)==self.cfg.goal: cell="goal"
                if (r,c) in self.cfg.holes: cell="hole"
                self.canvas.create_rectangle(x0,y0,x1,y1, fill=COLORS[cell], outline="")
                img = self.scaled_images.get(cell)
                if img is not None:
                    self.canvas.create_image(x0+CELL_SIZE//2, y0+CELL_SIZE//2, image=img)
        ar, ac = self.env.index_to_rc(self.env.state)
        agent_img = self.scaled_images.get("agent")
        if agent_img is not None:
            self.canvas.create_image(ac*CELL_SIZE+CELL_SIZE//2, ar*CELL_SIZE+CELL_SIZE//2, image=agent_img)
        else:
            m=ICON_MARGIN; self.canvas.create_oval(ac*CELL_SIZE+m, ar*CELL_SIZE+m, ac*CELL_SIZE+CELL_SIZE-2*m, ar*CELL_SIZE+CELL_SIZE-2*m, fill=COLORS['agent'], outline='black')
        for c in range(self.cfg.cols+1):
            x=c*CELL_SIZE; self.canvas.create_line(x,0,x,self.cfg.rows*CELL_SIZE, fill='black', width=1)
        for r in range(self.cfg.rows+1):
            y=r*CELL_SIZE; self.canvas.create_line(0,y,self.cfg.cols*CELL_SIZE,y, fill='black', width=1)
        if self.env.terminated:
            txt = "WIN" if self.env.index_to_rc(self.env.state)==self.cfg.goal else "FAIL"
            self.canvas.create_text(self.cfg.cols*CELL_SIZE//2, self.cfg.rows*CELL_SIZE//2, text=txt, font=("Arial",24), fill="red")

    def run_notebook(self):
        def worker():
            try:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
                nb_in = os.path.join(base_dir, "game_app", "notebook", "Frosen_POLICY_ITERATION_to_call.ipynb")
                if not os.path.exists(nb_in):
                    messagebox.showerror("Missing", f"Notebook not found:\n{nb_in}")
                    self.status_var.set("Notebook missing"); return
                out_dir = os.path.join(base_dir, "notebook"); os.makedirs(out_dir, exist_ok=True)
                nb_out = os.path.join(out_dir, "Frosen_POLICY_ITERATION_out.ipynb")
                py = sys.executable or "python"
                chk = subprocess.run([py, "-c", "import papermill"], cwd=base_dir)
                if chk.returncode != 0:
                    if not messagebox.askyesno("Papermill", "Papermill not installed. Install now?"):
                        self.status_var.set("Papermill missing"); return
                    self.status_var.set("Installing papermill…")
                    inst = subprocess.run([py, "-m", "pip", "install", "papermill"], cwd=base_dir, capture_output=True, text=True)
                    if inst.returncode != 0:
                        messagebox.showerror("Install Failed", inst.stderr or inst.stdout)
                        self.status_var.set("Install failed"); return
                rows=self.cfg.rows; cols=self.cfg.cols; sr,sc=self.cfg.start; gr,gc=self.cfg.goal
                holes=[(int(r),int(c)) for r,c in self.cfg.holes]
                # Build map description from current config
                grid=[["F" for _ in range(cols)] for _ in range(rows)]
                grid[sr][sc]="S"; grid[gr][gc]="G"
                for hr,hc in holes:
                    if (hr,hc) not in [(sr,sc),(gr,gc)]: grid[hr][hc]="H"
                map_desc=["".join(row) for row in grid]
                # Agent current state (may differ from start)
                cur_idx = int(self.env.state)
                cur_r, cur_c = self.env.index_to_rc(cur_idx)
                params={
                    "rows": int(rows),
                    "cols": int(cols),
                    "start": (int(sr),int(sc)),
                    "goal": (int(gr),int(gc)),
                    "holes": holes,
                    "slippery": bool(self.cfg.slippery),
                    "slip_prob": float(self.cfg.slip_prob),
                    "gamma": float(self.cfg.gamma),
                    "theta": float(self.cfg.theta),
                    "map_desc": map_desc,
                    "current_state": int(cur_idx),
                    "current_row": int(cur_r),
                    "current_col": int(cur_c),
                }
                # Create a JSON-friendly copy for papermill: tuples -> lists
                def _pm_convert(val):
                    if isinstance(val, tuple):
                        return list(val)
                    if isinstance(val, list):
                        new = []
                        for item in val:
                            if isinstance(item, tuple):
                                new.append(list(item))
                            else:
                                new.append(item)
                        return new
                    return val
                params_for_pm = {k: _pm_convert(v) for k, v in params.items()}
                import json
                param_file = os.path.join(base_dir, "game_app", "notebook", "params.json")
                with open(param_file, "w") as f:
                    json.dump(params_for_pm, f, indent=2)
                self.status_var.set(f"Running notebook with params:\n{params_for_pm}")
                import time
                time.sleep(1)
                # Prepare notebook: ensure it has a parameters cell (create temporary copy if needed)
                try:
                    import nbformat
                except Exception:
                    # try to install nbformat
                    subprocess.run([py, "-m", "pip", "install", "nbformat"], cwd=base_dir)
                    import nbformat
                nb = nbformat.read(nb_in, as_version=4)
                has_params = False
                for cell in nb.cells:
                    tags = cell.get('metadata', {}).get('tags', [])
                    if isinstance(tags, list) and 'parameters' in tags:
                        has_params = True
                        break
                nb_to_run = nb_in
                temp_nb = None
                if not has_params:
                    # Create a parameters cell at the top using Python literals (lists, numbers, bools)
                    def _literal(v):
                        # repr is fine for basic types (lists, numbers, booleans, strings)
                        return repr(v)
                    param_lines = [f"{k} = {_literal(v)}" for k, v in params_for_pm.items()]
                    param_src = "\n".join(param_lines)
                    new_cell = nbformat.v4.new_code_cell(param_src)
                    new_cell['metadata'] = new_cell.get('metadata', {})
                    new_cell['metadata'].setdefault('tags', [])
                    new_cell['metadata']['tags'].append('parameters')
                    nb.cells.insert(0, new_cell)
                    import tempfile
                    fd, temp_path = tempfile.mkstemp(prefix='pm_input_', suffix='.ipynb', dir=out_dir)
                    os.close(fd)
                    nbformat.write(nb, temp_path)
                    nb_to_run = temp_path
                    temp_nb = temp_path
                # Run the notebook with papermill using the chosen notebook file
                import papermill as pm
                pm.execute_notebook(nb_to_run, nb_out, parameters=params_for_pm)
                # remove temporary notebook if created
                try:
                    if temp_nb and os.path.exists(temp_nb):
                        os.remove(temp_nb)
                except Exception:
                    pass
                self.status_var.set(f"Notebook run complete:\n{nb_out}")
                # Optionally, open the output notebook
                if messagebox.askyesno("Open Notebook", "Open the output notebook?"):
                    if sys.platform == "win32":
                        os.startfile(nb_out)
                    elif sys.platform == "darwin":
                        subprocess.run(["open", nb_out])
                    else:
                        subprocess.run(["xdg-open", nb_out])
            except Exception as e:
                messagebox.showerror("Error", str(e))
        # Run the notebook worker in a separate thread to avoid blocking the UI
        threading.Thread(target=worker, daemon=True).start()

    def _update_status(self):
        # Update the status bar with current env configuration
        try:
            self.status_var.set(f"Slippery={'ON' if self.cfg.slippery else 'OFF'} p={self.cfg.slip_prob:.2f} Size={self.cfg.rows}x{self.cfg.cols}")
        except Exception:
            # Fallback simple status
            try:
                self.status_var.set("Ready")
            except Exception:
                pass

def main():
    root = tk.Tk(); FrozenLakeUI(root); root.mainloop()

if __name__ == "__main__":
    main()
