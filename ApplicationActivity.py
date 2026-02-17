import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Newton-Raphson Algorithm with Interactive Plotly Graphs ---
def newton_raphson_interactive(target_val, x0, tol=1e-6, max_iter=10):
    """
    Solves for sqrt(target_val) using Newton-Raphson on f(x) = x^2 - target_val
    """
    
    # Define function and derivative dynamically based on N
    def f(x): return x**2 - target_val
    def df(x): return 2*x

    iterations = []
    x_curr = x0
    
    # Create plot range centered on the guess and root
    x_range = np.linspace(0, x0 + 2, 1000)
    y_range = f(x_range)
    
    print(f"\n--- Solving for sqrt({target_val}) with x0={x0} ---")
    print(f"{'Iter':<5} {'x_curr':<12} {'f(x)':<12} {'Error':<12}")
    
    # Store frames for the interactive graph
    fig = go.Figure()

    # Add the main curve (Parabola)
    fig.add_trace(go.Scatter(
        x=x_range, y=y_range,
        mode='lines', name=f'f(x) = x^2 - {target_val}',
        line=dict(color='blue', width=2)
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)

    for i in range(max_iter):
        fx = f(x_curr)
        dfx = df(x_curr)
        
        if abs(dfx) < 1e-12: break # Avoid division by zero
            
        x_next = x_curr - fx/dfx
        error = abs(x_next - x_curr)
        
        # --- Visualization: Add Tangent Lines & Points ---
        # Tangent line segment from (x_curr, fx) to (x_next, 0)
        fig.add_trace(go.Scatter(
            x=[x_curr, x_next], y=[fx, 0],
            mode='lines+markers',
            name=f'Iter {i+1} Tangent',
            line=dict(width=1.5, dash='dash'),
            marker=dict(size=6),
            visible=True # All visible to show history, or set 'legendonly' to toggle
        ))
        
        # Mark the specific points
        fig.add_trace(go.Scatter(
            x=[x_curr], y=[fx],
            mode='markers',
            name=f'Start {i+1}: {x_curr:.4f}',
            marker=dict(color='green', size=8)
        ))
        # -------------------------------------------------

        iterations.append({'Iter': i+1, 'x_curr': x_curr, 'f(x)': fx, 'Error': error})
        print(f"{i+1:<5} {x_curr:<12.6f} {fx:<12.6f} {error:<12.6f}")

        if error < tol:
            print(f"Converged to {x_next:.6f}")
            break
        
        x_curr = x_next

    # --- Final Plot Layout Settings (Zoom/Pan Enabled) ---
    fig.update_layout(
        title=f"Newton-Raphson for √{target_val} (Final Root: {x_curr:.5f})",
        xaxis_title="x",
        yaxis_title="f(x) = x² - N",
        template="plotly_white",
        dragmode='pan',   # Allows clicking and dragging
        height=600,
        hovermode="x unified"
    )
    # Enable scroll zoom
    fig.show(config={'scrollZoom': True, 'displayModeBar': True})
    
    return pd.DataFrame(iterations)

# --- 2. Execute for Handwritten Note Problems ---
problems = [
    (3, 2.0),   # sqrt(3), guess 2
    (7, 3.0),   # sqrt(7), guess 3
    (20, 4.0),  # sqrt(20), guess 4
    (99, 10.0)  # sqrt(99), guess 10
]

for N, guess in problems:
    newton_raphson_interactive(N, guess)
