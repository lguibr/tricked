#!/usr/bin/env python3
import os
import optuna
import optunahub


def generate_insights():
    db_path = "sqlite:///autotune.db"
    study_name = "tricked-ai-tuning-sota-2h"

    print(f"🔍 Loading Study '{study_name}'...")
    try:
        study = optuna.load_study(study_name=study_name, storage=db_path)
        print(f"✅ Loaded {len(study.trials)} trials.")
    except Exception as e:
        print(f"❌ Could not load study: {e}")
        return

    output_dir = "outputs/insights"
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Parallel Coordinate Plot (Built-In)
    # The ultimate high-dimensional flow visualizer.
    # ---------------------------------------------------------
    print("📈 Generating Parallel Coordinate Plot...")
    fig_parallel = optuna.visualization.plot_parallel_coordinate(
        study,
        params=[
            "simulations",
            "gumbel_scale",
            "unroll_steps",
            "temporal_difference_steps",
            "lr_init",
        ],
    )
    fig_parallel.update_layout(
        title="High-Dimensional Flow: How parameters route to the highest score",
        template="plotly_dark",
    )
    fig_parallel.write_html(f"{output_dir}/1_parallel_coordinates.html")

    # ---------------------------------------------------------
    # 2. Contour Plot (Built-In)
    # The 2D Heatmap for specific interaction boundaries.
    # ---------------------------------------------------------
    print("🗺️ Generating Contour Heatmaps...")
    # Plot Mcripts Depth vs Exploration Noise
    fig_contour_mcts = optuna.visualization.plot_contour(
        study, params=["simulations", "gumbel_scale"]
    )
    fig_contour_mcts.update_layout(
        title="Contour: Simulations vs Gumbel Scale", template="plotly_dark"
    )
    # Plot Bootstrapping Horizon vs Dynamics Prediction
    fig_contour_mcts.write_html(f"{output_dir}/2_contour_mcts_exploration.html")

    fig_contour_td = optuna.visualization.plot_contour(
        study, params=["temporal_difference_steps", "unroll_steps"]
    )
    fig_contour_td.update_layout(
        title="Contour: TD Steps vs Unroll Steps", template="plotly_dark"
    )
    fig_contour_td.write_html(f"{output_dir}/3_contour_value_horizon.html")

    # ---------------------------------------------------------
    # 3. SHAP-like Beeswarm Plot (OptunaHub)
    # Shows if a high parameter value pushes the score up or down.
    # ---------------------------------------------------------
    print("🐝 Loading SHAP-like Beeswarm from OptunaHub...")
    try:
        # Note: optunahub searches registered namespaces. We use the standard import pattern.
        shap_module = optunahub.load_module("visualization/plot_beeswarm")
        fig, ax, cbar = shap_module.plot_beeswarm(study)
        fig.suptitle("SHAP Beeswarm: Directional Parameter Impact")
        fig.savefig(f"{output_dir}/4_shap_beeswarm.png")
        print("✅ Saved 4_shap_beeswarm.png (Matplotlib)")
    except Exception as e:
        print(f"⚠️ Could not load/render SHAP Beeswarm from OptunaHub: {e}")
        # Fallback to Built-in Importance if Hub fails
        print("Falling back to Built-In Hyperparameter Importances...")
        fig_import = optuna.visualization.plot_param_importances(study)
        fig_import.update_layout(
            title="fANOVA Parameter Importances", template="plotly_dark"
        )
        fig_import.write_html(f"{output_dir}/4_basic_importance.html")

    # ---------------------------------------------------------
    # 4. Step Distribution Plot (OptunaHub)
    # Shows exactly when and where the pruner is killing bad trials.
    # ---------------------------------------------------------
    print("📉 Loading Step Distribution from OptunaHub...")
    fig_step = None
    try:
        step_dist_module = optunahub.load_module("visualization/plot_step_distribution")
        fig_step = step_dist_module.plot_step_distribution(study)
        fig_step.update_layout(
            title="Pruning Step Distribution (Where agents die)", template="plotly_dark"
        )
        fig_step.write_html(f"{output_dir}/5_step_distribution.html")
    except Exception as e:
        print(f"⚠️ Could not load/render Step Distribution from OptunaHub: {e}")
        # Fallback to Built-in Optimization History
        print("Falling back to Built-in Optimization History...")
        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.update_layout(title="Optimization History", template="plotly_dark")
        fig_history.write_html(f"{output_dir}/5_optimization_history.html")

    # ---------------------------------------------------------
    # Dashboard Integration
    # ---------------------------------------------------------
    try:
        from optuna_dashboard import save_plotly_graph_object

        print("💻 Pushing Plotly graphs to Optuna Dashboard...")
        save_plotly_graph_object(study, fig_parallel)
        save_plotly_graph_object(study, fig_contour_mcts)
        save_plotly_graph_object(study, fig_contour_td)
        if fig_step is not None:
            save_plotly_graph_object(study, fig_step)
        print("✅ Custom Plotly graphs are now visible in Optuna Dashboard!")
    except ImportError:
        print("⚠️ optuna-dashboard not installed. Skipping dashboard push.")
    except Exception as e:
        print(f"⚠️ Could not push to dashboard: {e}")

    print(
        f"\n🎉 Success! Open the files in '{output_dir}' to view the interactive insights standalone, or check your Optuna Dashboard!"
    )


if __name__ == "__main__":
    generate_insights()
