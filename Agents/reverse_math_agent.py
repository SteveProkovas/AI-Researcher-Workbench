import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import random
from sympy import symbols, Eq, solve, sympify, expand, Poly
from sympy.parsing.sympy_parser import parse_expr
from fractions import Fraction

# --- Helper generators for templates ---

x, y = symbols('x y')

def parse_solution(text):
    """
    Parse user text into a sympy object representing the solution.
    Accepts forms:
      - a single numeric: "3", "1/2"
      - an equation form: "x=3"
      - tuple/list: "x=3,y=2" or "(3,2)"
    Returns a dict like {'x': value} or {'': numeric} for single unnamed numeric.
    """
    text = text.strip()
    # Try "x=3" case
    if '=' in text and ',' in text:
        parts = [p.strip() for p in text.split(',')]
        sol = {}
        for p in parts:
            k, v = p.split('=')
            sol[k.strip()] = sympify(v.strip())
        return sol
    if '=' in text:
        k, v = text.split('=')
        return {k.strip(): sympify(v.strip())}
    # no variable given -> treat as numeric target
    try:
        val = sympify(text)
        return {'': val}
    except Exception:
        # fallback try parse_expr
        return {'': parse_expr(text)}

def generate_linear_from_solution(sol):
    """
    Generate linear equation problem that matches solution.
    For solution x = s, produce equation: a*x + b = 0 with integer a and b so that x = s
    We'll choose a as a small non-zero integer and set b = -a*s.
    Returns (problem_expr, eq_text, test_functions)
    """
    # Expect sol like {'x': s} or {'': s}
    if 'x' in sol:
        s = sol['x']
    elif '' in sol:
        s = sol['']
    else:
        # pick first variable
        k = next(iter(sol))
        s = sol[k]

    # choose integer a (avoid 0)
    a = random.choice([i for i in range(-5, 6) if i != 0])
    # b must be -a*s. To keep integer b, try to convert s to Fraction if numeric
    try:
        b = -a * sympify(s)
    except Exception:
        b = -a * s

    # create equation a*x + b = 0
    eq = Eq(a * x + b, 0)
    text = f"Solve for x: {a}·x + ({b}) = 0"
    # test: solve(eq) == s
    def verify():
        sols = solve(eq, x)
        return sols and (sympify(s) in sols)

    return {'type': 'Linear equation', 'equation': eq, 'text': text, 'verify': verify}

def generate_quadratic_with_root(sol):
    """
    Create quadratic (x - r1)(x - r2) = 0 with r1 = given solution.
    Choose integer r2 randomly.
    """
    if 'x' in sol:
        r1 = sol['x']
    elif '' in sol:
        r1 = sol['']
    else:
        r1 = next(iter(sol.values()))
    # pick integer r2 (not equal to r1 if possible)
    r2_candidates = [i for i in range(-5, 6)]
    # if r1 is numeric and integer-ish, avoid equal
    r2 = random.choice(r2_candidates)
    # expand
    expr = expand((x - r1) * (x - r2))
    eq = Eq(expr, 0)
    text = f"Solve for x: {Poly(expr, x).as_expr()} = 0"
    def verify():
        sols = solve(eq, x)
        return any(sympify(r1) == soln for soln in sols)
    return {'type': 'Quadratic equation', 'equation': eq, 'text': text, 'verify': verify}

def generate_system_from_solution(sol):
    """
    If solution contains both x and y, create a simple 2x2 linear system with integer coefficients
    that yields those values exactly.
    We'll construct:
       a*x + b*y = c
       d*x + e*y = f
    Solve for a..f integers so that solution matches.
    """
    if 'x' not in sol or 'y' not in sol:
        raise ValueError("System generator expects solution dict with 'x' and 'y'")
    rx = sol['x']
    ry = sol['y']

    # choose integer matrix coeffs
    a, b = random.randint(1,5), random.randint(1,5)
    d, e = random.randint(1,5), random.randint(1,5)
    # compute c = a*rx + b*ry etc.
    c = a*rx + b*ry
    f = d*rx + e*ry
    eq1 = Eq(a*x + b*y, c)
    eq2 = Eq(d*x + e*y, f)
    text = f"Solve the system:\n{a}x + {b}y = {c}\n{d}x + {e}y = {f}"
    def verify():
        sols = solve((eq1, eq2), (x,y))
        return sols and sols.get(x)==sympify(rx) and sols.get(y)==sympify(ry)
    return {'type':'2x2 linear system', 'equation': (eq1, eq2), 'text': text, 'verify': verify}

# map template name -> generator
TEMPLATES = {
    "Linear (solve for x)": generate_linear_from_solution,
    "Quadratic (root = solution)": generate_quadratic_with_root,
    "2x2 Linear System (x,y)": generate_system_from_solution
}

# --- GUI ---

class ReverseAgentUI:
    def __init__(self, root):
        self.root = root
        root.title("Reverse-Order Math Problem Generator")
        self.mainframe = ttk.Frame(root, padding=12)
        self.mainframe.grid(row=0, column=0, sticky="nsew")

        ttk.Label(self.mainframe, text="Enter target solution (e.g. x=3 or x=2,y=5 or 3):").grid(row=0, column=0, sticky="w")
        self.solution_entry = ttk.Entry(self.mainframe, width=40)
        self.solution_entry.insert(0, "x=3")
        self.solution_entry.grid(row=1, column=0, sticky="w")

        ttk.Label(self.mainframe, text="Template:").grid(row=2, column=0, sticky="w")
        self.template_var = tk.StringVar(value=list(TEMPLATES.keys())[0])
        self.template_menu = ttk.OptionMenu(self.mainframe, self.template_var, self.template_var.get(), *TEMPLATES.keys())
        self.template_menu.grid(row=3, column=0, sticky="w")

        self.generate_btn = ttk.Button(self.mainframe, text="Generate problem & tests", command=self.generate)
        self.generate_btn.grid(row=4, column=0, pady=(8,0), sticky="w")

        ttk.Label(self.mainframe, text="Generated problem:").grid(row=5, column=0, sticky="w", pady=(10,0))
        self.problem_box = scrolledtext.ScrolledText(self.mainframe, width=60, height=6)
        self.problem_box.grid(row=6, column=0)

        self.verify_btn = ttk.Button(self.mainframe, text="Run verification tests", command=self.run_verify)
        self.verify_btn.grid(row=7, column=0, sticky="w", pady=(8,0))

        ttk.Label(self.mainframe, text="Verification log:").grid(row=8, column=0, sticky="w", pady=(10,0))
        self.log_box = scrolledtext.ScrolledText(self.mainframe, width=60, height=8)
        self.log_box.grid(row=9, column=0)

        # hold last generated artifact
        self.last_artifact = None

    def log(self, s):
        self.log_box.insert(tk.END, s+"\n")
        self.log_box.see(tk.END)

    def generate(self):
        self.log_box.delete('1.0', tk.END)
        self.problem_box.delete('1.0', tk.END)
        sol_text = self.solution_entry.get()
        try:
            sol = parse_solution(sol_text)
        except Exception as e:
            messagebox.showerror("Parse error", f"Could not parse solution: {e}")
            return
        template_name = self.template_var.get()
        gen = TEMPLATES.get(template_name)
        try:
            artifact = gen(sol)
        except Exception as e:
            messagebox.showerror("Generation error", str(e))
            return
        # store artifact
        self.last_artifact = (artifact, sol)
        # show text
        self.problem_box.insert(tk.END, f"Template: {artifact['type']}\n\n{artifact['text']}\n\nGenerated tests will symbolically verify the solution.")
        self.log(f"Generated {artifact['type']}. Ready to verify.")

    def run_verify(self):
        if not self.last_artifact:
            messagebox.showinfo("Nothing to verify", "Generate a problem first.")
            return
        artifact, sol = self.last_artifact
        self.log_box.delete('1.0', tk.END)
        self.log("Running verification...")
        ok = False
        try:
            ok = artifact['verify']()
        except Exception as e:
            self.log(f"Error during verification: {e}")
            ok = False
        self.log(f"Verification result: {'PASS' if ok else 'FAIL'}")
        if ok:
            self.log("The generated problem solves to the requested solution.")
        else:
            self.log("Mismatch: the problem does not solve to the requested solution (or generator unsupported).")

def main():
    root = tk.Tk()
    ReverseAgentUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
