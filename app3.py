# Import the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch

###############################
#   Function Definitions
###############################

def analyze_excel_and_generate_tables(file_path, sheet_name=0):
    """
    Analyzes the specified Excel file and sheet, identifying scenarios, contributions,
    and generating initial and combined tables for impact assessment.
    """
    try:
        # Read the Excel file
        data = pd.read_excel(file_path, header=None, sheet_name=sheet_name)
        print(f"Excel file loaded successfully from sheet: {sheet_name}!")

        # Step 1: Detect scenarios
        scenario_row = data.iloc[0, 1:]
        scenario_names = scenario_row.unique()
        scenario_names_cleaned = [
            name[name.find("(") + 1 : name.find(")")].strip().lower()
            for name in scenario_names
        ]
        print(f"Detected scenarios ({len(scenario_names_cleaned)}): {scenario_names_cleaned}")

        # Step 2: Detect contributions
        contribution_row = data.iloc[1, 1:]
        print(f"Detected contributions: {list(contribution_row.unique())}")

        # Step 3: Structure the data
        categories = data.iloc[2:, 0]
        values = data.iloc[2:, 1:]
        values.columns = pd.MultiIndex.from_arrays([scenario_row, contribution_row])
        values.index = categories

        # Initial table
        initial_table = values.copy()

        # Impact totals (absolute values)
        total_impact_table = values.abs().T.groupby(level=0).sum().T

        # Replace columns with simplified scenario names
        total_impact_table.columns = [
            name[name.find("(") + 1 : name.find(")")].strip()
            for name in total_impact_table.columns
        ]

        # Relative percentages
        relative_percentage_table = total_impact_table.copy()
        for category in relative_percentage_table.index:
            max_value = relative_percentage_table.loc[category].max()
            relative_percentage_table.loc[category] = (
                relative_percentage_table.loc[category] / max_value
            ) * 100
        relative_percentage_table = relative_percentage_table.add_suffix(" (%)")

        # Combine tables
        combined_table = pd.concat([total_impact_table, relative_percentage_table], axis=1)

        return initial_table, combined_table, total_impact_table, scenario_names_cleaned
    except Exception as e:
        print(f"Error analyzing the Excel file: {e}")
        return None, None, None, None

def extract_contributions_from_initial_table(initial_table):
    """
    Extrait les contributions uniques depuis les colonnes MultiIndex de initial_table.
    """
    try:
        if not isinstance(initial_table.columns, pd.MultiIndex):
            raise ValueError("initial_table n'a pas un MultiIndex en colonnes.")

        contributions = initial_table.columns.get_level_values(1)
        unique_contributions = sorted(set(contributions.astype(str).str.strip()))

        print(f"Contributions uniques détectées ({len(unique_contributions)}) : {unique_contributions}")
        return unique_contributions

    except Exception as e:
        print(f"Erreur lors de l'extraction depuis initial_table : {e}")
        return []
    
def generate_percentage_table(initial_table, total_impact_table):
    """
    Generates a DataFrame identical to `initial_table`, but converts its values to percentages
    relative to the maximum impact found in `total_impact_table` for each category.
    """
    try:
        percentage_table = initial_table.copy()
        
        for category in initial_table.index:
            # Get the maximum impact for this category
            max_total_impact = total_impact_table.loc[category].max()
            # Calculate percentages
            percentage_table.loc[category] = (
                initial_table.loc[category] / max_total_impact
            ) * 100

        return percentage_table
    except Exception as e:
        print(f"Error generating percentage table: {e}")
        return None


def generate_tables_by_scenario(initial_table, scenario_names_cleaned):
    """
    Creates a dictionary of tables, one per scenario, containing:
    - Impact categories
    - Contributions
    - Total impact per category
    - Percentage of each contribution relative to the total impact
    """
    try:
        scenario_tables = {}
        for scenario_clean in scenario_names_cleaned:
            # Find the matching scenario in the MultiIndex columns
            matching_scenario = [
                col for col in initial_table.columns.levels[0] if scenario_clean in col.lower()
            ]
            if not matching_scenario:
                raise KeyError(f"No column match found for simplified scenario '{scenario_clean}'.")
            matching_scenario = matching_scenario[0]

            # Filter data for this scenario
            scenario_data = initial_table.xs(key=matching_scenario, axis=1, level=0)

            # Create a formatted table
            scenario_table = pd.DataFrame()
            scenario_table["Impact Category"] = initial_table.index
            for contribution in scenario_data.columns:
                scenario_table[contribution] = scenario_data[contribution].values

            # Calculate total impact (absolute values)
            scenario_table["Total Impact"] = scenario_table.iloc[:, 1:].abs().sum(axis=1)

            # Calculate contribution percentages
            for contribution in scenario_data.columns:
                scenario_table[f"% {contribution}"] = (
                    scenario_table[contribution] / scenario_table["Total Impact"] * 100
                )

            scenario_tables[scenario_clean] = scenario_table

            print(f"\nGenerated table for scenario: {scenario_clean}")
            print(scenario_table.head())

        return scenario_tables
    except Exception as e:
        print(f"Error generating scenario tables: {e}")
        return None


def plot_comparison_bar_chart(total_impact_table):
    """
    Plots 3 figures:
    1) Main comparison bar chart
    2) Separate legend
    3) Bar chart with total values labeled
    Returns a list of matplotlib figures.
    """
    try:
        global scenario_colors
        figs = []

        # Normalize data
        normalized_data = total_impact_table.copy()
        for category in normalized_data.index:
            max_value = normalized_data.loc[category].max()
            normalized_data.loc[category] = (normalized_data.loc[category] / max_value) * 100

        categories = normalized_data.index
        scenarios = normalized_data.columns
        num_scenarios = len(scenarios)
        bar_width = 0.9 / num_scenarios
        x_positions = np.arange(len(categories))

        # === Handle scenario colors ===
        scenario_color_map = None
        generic_colors = plt.get_cmap("tab10").colors

        if scenario_colors is not None:
            if isinstance(scenario_colors, dict):
                scenario_color_map = scenario_colors
            elif hasattr(scenario_colors, "iterrows"):
                scenario_color_map = {
                    row["Scenario"].strip().lower(): row["Hex Code"]
                    for _, row in scenario_colors.iterrows()
                }

        # === Figure 1: Main bar chart (no totals) ===
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        bars = []
        labels = []

        for i, scenario in enumerate(scenarios):
            key = scenario.strip().lower()
            color = (
                scenario_color_map.get(key, "#000000")
                if scenario_color_map else generic_colors[i % len(generic_colors)]
            )
            bar = ax1.bar(
                x_positions + i * bar_width,
                normalized_data[scenario],
                bar_width,
                color=color,
            )
            bars.append(bar[0])
            labels.append(scenario)

        ax1.set_title("Comparison of Scenarios", fontsize=16, pad=30, fontweight="bold")
        ax1.set_ylabel("(%)", fontsize=14)
        ax1.set_ylim(0, 100)
        ax1.set_xlim(-0.5, len(categories))
        ax1.set_xticks(x_positions + (len(scenarios) - 1) * bar_width / 2)
        ax1.set_xticklabels(categories, rotation=45, ha="right", fontsize=12)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        figs.append(fig1)
        plt.close(fig1)

        # === Figure 2: Legend only ===
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.axis("off")
        ax2.legend(bars, labels, title="Scenarios", fontsize=10, loc="center")
        plt.tight_layout()
        figs.append(fig2)
        plt.close(fig2)

        # === Figure 3: With total values ===
        fig3, ax3 = plt.subplots(figsize=(12, 7))
        for i, scenario in enumerate(scenarios):
            key = scenario.strip().lower()
            color = (
                scenario_color_map.get(key, "#000000")
                if scenario_color_map else generic_colors[i % len(generic_colors)]
            )
            ax3.bar(
                x_positions + i * bar_width,
                normalized_data[scenario],
                bar_width,
                color=color,
            )

        for i, scenario in enumerate(scenarios):
            for j, value in enumerate(normalized_data[scenario]):
                total_value = total_impact_table.loc[categories[j], scenario]
                if 0.01 <= abs(total_value) <= 1000:
                    formatted_value = f"{total_value:.2f}"
                else:
                    formatted_value = f"{total_value:.2E}"
                ax3.text(
                    x_positions[j] + i * bar_width,
                    value + 2,
                    formatted_value,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="black",
                    fontweight="bold",
                    rotation=90 if abs(total_value) < 0.01 else 0,
                )

        ax3.set_title("Comparison of Scenarios (with Totals)", fontsize=16, pad=80, fontweight="bold")
        ax3.set_ylabel("(%)", fontsize=14)
        ax3.set_ylim(0, 100)
        ax3.set_xlim(-0.5, len(categories))
        ax3.set_xticks(x_positions + (len(scenarios) - 1) * bar_width / 2)
        ax3.set_xticklabels(categories, rotation=45, ha="right", fontsize=12)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        figs.append(fig3)
        plt.close(fig3)

        return figs

    except Exception as e:
        st.error(f"Error generating comparison bar chart: {str(e)}")
        return []



def plot_relative_contribution_by_scenario(scenario_tables):
    """
    Génère une liste de (main_figure, legend_figure) pour chaque scénario,
    affichant un graphique en barres empilées (verticales) des contributions relatives.
    """
    try:
        global contributions_colors
        figures = []

        # Identifier toutes les contributions présentes
        all_contributions = set()
        for table in scenario_tables.values():
            contribution_columns = [
                col for col in table.columns
                if col.startswith("%") and col != "%Total Impact"
            ]
            all_contributions.update(contribution_columns)

        # Générer la palette de couleurs
        contribution_colors = {}
        generic_colors = plt.get_cmap("tab10").colors

        if contributions_colors is not None:
            if isinstance(contributions_colors, dict):
                for i, contrib in enumerate(sorted(all_contributions)):
                    key = contrib.replace("%", "").strip().lower()
                    contribution_colors[contrib] = contributions_colors.get(key, generic_colors[i % len(generic_colors)])
            elif hasattr(contributions_colors, "iterrows"):
                for i, contrib in enumerate(sorted(all_contributions)):
                    contrib_clean = contrib.replace("%", "").strip().lower()
                    match = contributions_colors[
                        contributions_colors["Contributions"].str.lower() == contrib_clean
                    ]
                    if not match.empty:
                        contribution_colors[contrib] = match["Hex Code"].iloc[0]
                    else:
                        contribution_colors[contrib] = generic_colors[i % len(generic_colors)]
        else:
            for i, contrib in enumerate(sorted(all_contributions)):
                contribution_colors[contrib] = generic_colors[i % len(generic_colors)]

        # Génération des graphiques
        for scenario_name, table in scenario_tables.items():
            main_fig, main_ax = plt.subplots(figsize=(16, 8))
            legend_fig, legend_ax = plt.subplots(figsize=(5, 3))
            
            categories = table["Impact Category"]
            contribution_columns = [
                col for col in table.columns
                if col.startswith("%") and col != "%Total Impact"
            ]
            percentage_data = table[contribution_columns].astype(float)
            total_impact = table["Total Impact"].astype(float)
            bar_width = 0.6
            bars = []
            labels = []

            bottom_pos = np.zeros(len(categories))
            bottom_neg = np.zeros(len(categories))

            for contribution in contribution_columns:
                values = percentage_data[contribution].values
                pos_values = np.where(values > 0, values, 0)
                neg_values = np.where(values < 0, values, 0)

                if np.any(pos_values):
                    bar = main_ax.bar(
                        categories, pos_values, bar_width,
                        bottom=bottom_pos,
                        color=contribution_colors[contribution]
                    )
                    bars.append(bar[0])
                    labels.append(contribution.replace("%", ""))
                    bottom_pos += pos_values

                if np.any(neg_values):
                    main_ax.bar(
                        categories, neg_values, bar_width,
                        bottom=bottom_neg,
                        color=contribution_colors[contribution]
                    )
                    bottom_neg += neg_values

            main_ax.axhline(0, color="black", linestyle="--", alpha=0.7)

            # Ajouter les valeurs totales au-dessus
            for i, total in enumerate(total_impact):
                formatted_value = f"{total:.2f}" if abs(total) <= 1000 else f"{total:.2E}"
                main_ax.text(
                    i, max(bottom_pos[i], 0) + 1,
                    formatted_value,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="black",
                    fontweight="bold"
                )

            main_ax.set_title(
                f"Relative Contributions - {scenario_name.capitalize()}",
                fontsize=16,
                pad=40,
                fontweight="bold"
            )
            main_ax.set_ylabel("(%)")
            main_ax.set_ylim(min(bottom_neg) - 5, max(bottom_pos))
            main_ax.set_xticks(range(len(categories)))
            main_ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=12)
            main_ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            legend_ax.axis("off")
            legend_ax.legend(bars, labels, title="Contributions", fontsize=10, loc="center")
            plt.tight_layout()

            figures.append((main_fig, legend_fig))
            plt.close(main_fig)
            plt.close(legend_fig)

        return figures

    except Exception as e:
        st.error(f"Error generating relative contributions: {str(e)}")
        return []




def plot_relative_contribution_by_scenario_horizontal(scenario_tables):
    """
    Génère des (main_figure, legend_figure) pour chaque scénario
    affichant un bar chart horizontal empilé des contributions relatives.
    """
    try:
        global contributions_colors
        figures = []

        # Identifier toutes les contributions
        all_contributions = set()
        for table in scenario_tables.values():
            contribution_columns = [
                col for col in table.columns
                if col.startswith("%") and col != "%Total Impact"
            ]
            all_contributions.update(contribution_columns)

        # Générer la palette de couleurs
        contribution_color_map = {}
        generic_colors = plt.get_cmap("tab10").colors

        if contributions_colors is not None:
            if isinstance(contributions_colors, dict):
                for i, contrib in enumerate(sorted(all_contributions)):
                    key = contrib.replace("%", "").strip().lower()
                    contribution_color_map[contrib] = contributions_colors.get(key, generic_colors[i % len(generic_colors)])
            elif hasattr(contributions_colors, "iterrows"):
                for i, contrib in enumerate(sorted(all_contributions)):
                    contrib_clean = contrib.replace("%", "").strip().lower()
                    match = contributions_colors[
                        contributions_colors["Contributions"].str.lower() == contrib_clean
                    ]
                    if not match.empty:
                        contribution_color_map[contrib] = match["Hex Code"].iloc[0]
                    else:
                        contribution_color_map[contrib] = generic_colors[i % len(generic_colors)]
        else:
            for i, contrib in enumerate(sorted(all_contributions)):
                contribution_color_map[contrib] = generic_colors[i % len(generic_colors)]

        # Génération des graphiques
        for scenario_name, table in scenario_tables.items():
            main_fig, main_ax = plt.subplots(figsize=(10, 12))
            legend_fig, legend_ax = plt.subplots(figsize=(5, 3))

            categories = table["Impact Category"]
            contribution_columns = [
                col for col in table.columns
                if col.startswith("%") and col != "%Total Impact"
            ]
            percentage_data = table[contribution_columns].astype(float)
            total_impact = table["Total Impact"].astype(float)
            bar_height = 0.6
            bars = []
            labels = []

            left_pos = np.zeros(len(categories))
            left_neg = np.zeros(len(categories))

            for contribution in contribution_columns:
                values = percentage_data[contribution].values
                pos_values = np.where(values > 0, values, 0)
                neg_values = np.where(values < 0, values, 0)

                if np.any(pos_values):
                    bar = main_ax.barh(
                        categories,
                        pos_values,
                        bar_height,
                        left=left_pos,
                        color=contribution_color_map[contribution]
                    )
                    bars.append(bar[0])
                    labels.append(contribution.replace("%", ""))
                    left_pos += pos_values

                if np.any(neg_values):
                    main_ax.barh(
                        categories,
                        neg_values,
                        bar_height,
                        left=left_neg,
                        color=contribution_color_map[contribution]
                    )
                    left_neg += neg_values

            main_ax.axvline(0, color="black", linestyle="--", alpha=0.7)

            # Affichage des valeurs totales à droite
            for i, total in enumerate(total_impact):
                formatted_value = f"{total:.2f}" if abs(total) <= 1000 else f"{total:.2E}"
                main_ax.text(
                    max(left_pos[i], 0) + 5,
                    i,
                    formatted_value,
                    ha="left",
                    va="center",
                    fontsize=10,
                    color="black",
                    fontweight="bold"
                )

            main_ax.set_title(
                f"Horizontal Contributions - {scenario_name.capitalize()}",
                fontsize=16,
                pad=30,
                fontweight="bold"
            )
            main_ax.set_xlabel("(%)")
            main_ax.set_xlim(min(left_neg) - 5, max(left_pos))
            main_ax.set_yticks(range(len(categories)))
            main_ax.set_yticklabels(categories, fontsize=12)
            main_ax.grid(axis="x", linestyle="--", alpha=0.7)
            plt.tight_layout()

            legend_ax.axis("off")
            legend_ax.legend(bars, labels, title="Contributions", fontsize=10, loc="center")
            plt.tight_layout()

            figures.append((main_fig, legend_fig))
            plt.close(main_fig)
            plt.close(legend_fig)

        return figures

    except Exception as e:
        st.error(f"Error generating horizontal contributions: {str(e)}")
        return []



def plot_stacked_bar_by_category(percentage_table, total_impact_table):
    """
    Génère (main_figure, legend_figure, category_name) pour chaque catégorie d’impact.
    Adapté pour affichage dans Streamlit via st.pyplot().
    """
    try:
        
        global contributions_colors

        figures = []

        if percentage_table.empty or total_impact_table.empty:
            st.warning("⚠ Données vides pour les graphiques par catégorie.")
            return []

        # 🔹 Obtenir les scénarios et contributions
        scenarios = percentage_table.columns.get_level_values(0).unique()
        contributions = percentage_table.columns.get_level_values(1).unique()
        categories = percentage_table.index

        # 🔹 Nettoyage des noms de scénarios
        scenario_labels = {scenario: scenario.split("(")[-1].replace(")", "").strip().lower() for scenario in scenarios}
        total_impact_labels = {sc.lower(): sc for sc in total_impact_table.columns}

        # 🔹 Générer le mapping de couleurs des contributions
        contrib_color_map = {}
        generic_colors = plt.get_cmap("tab10").colors
        if contributions_colors is not None:
            if isinstance(contributions_colors, dict):
                for i, contrib in enumerate(contributions):
                    key = contrib.replace("%", "").strip().lower()
                    contrib_color_map[contrib] = contributions_colors.get(key, generic_colors[i % len(generic_colors)])
            elif hasattr(contributions_colors, "iterrows"):
                for i, contrib in enumerate(contributions):
                    contrib_clean = contrib.replace("%", "").strip().lower()
                    match = contributions_colors[contributions_colors["Contributions"].str.lower() == contrib_clean]
                    if not match.empty:
                        contrib_color_map[contrib] = match["Hex Code"].iloc[0]
                    else:
                        contrib_color_map[contrib] = generic_colors[i % len(generic_colors)]
        else:
            for i, contrib in enumerate(contributions):
                contrib_color_map[contrib] = generic_colors[i % len(generic_colors)]

        # 🔹 Générer les graphiques pour chaque catégorie
        for category in categories:
            category_data = percentage_table.loc[category]
            x_positions = np.arange(len(scenarios))
            bar_width = max(0.2, min(0.8, 1.5 / len(scenarios)))

            main_fig, main_ax = plt.subplots(figsize=(12, 7))
            legend_fig, legend_ax = plt.subplots(figsize=(5, 3))

            bottom_pos = np.zeros(len(scenarios))
            bottom_neg = np.zeros(len(scenarios))
            bars = []
            labels = []

            for contribution in contributions:
                values = []
                for s in scenarios:
                    try:
                        values.append(category_data[s, contribution])
                    except KeyError:
                        values.append(0)
                values = np.array(values, dtype=float)

                pos_values = np.where(values > 0, values, 0)
                neg_values = np.where(values < 0, values, 0)

                if np.any(pos_values):
                    bar = main_ax.bar(
                        x_positions,
                        pos_values,
                        width=bar_width,
                        bottom=bottom_pos,
                        color=contrib_color_map.get(contribution, "#000000")
                    )
                    bars.append(bar[0])
                    labels.append(contribution.replace("%", ""))
                    bottom_pos += pos_values

                if np.any(neg_values):
                    main_ax.bar(
                        x_positions,
                        neg_values,
                        width=bar_width,
                        bottom=bottom_neg,
                        color=contrib_color_map.get(contribution, "#000000")
                    )
                    bottom_neg += neg_values

            # Ajouter les totaux
            for i, scenario in enumerate(scenarios):
                scen_clean = scenario_labels[scenario]
                if scen_clean in total_impact_labels:
                    try:
                        total_value = total_impact_table.loc[category, total_impact_labels[scen_clean]]
                        formatted = f"{total_value:.2f}" if 0.01 <= abs(total_value) <= 1000 else f"{total_value:.2E}"
                        main_ax.text(
                            x_positions[i],
                            max(bottom_pos[i], 0) + 5,
                            formatted,
                            ha='center',
                            va='bottom',
                            fontsize=10,
                            color='black',
                            fontweight='bold'
                        )
                    except KeyError:
                        continue

            # Mise en forme
            main_ax.set_title(f"Contribution Analysis - Category: {category}", fontsize=16, pad=40, fontweight="bold")
            main_ax.set_ylabel("(%)", fontsize=14)
            main_ax.set_xticks(x_positions)
            main_ax.set_xticklabels([scenario_labels[s] for s in scenarios], rotation=45, ha="right", fontsize=12)
            main_ax.set_xlim(x_positions[0] - (bar_width / 2), x_positions[-1] + (bar_width / 2))
            main_ax.set_ylim(min(bottom_neg) - 5, max(bottom_pos))
            main_ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            legend_ax.axis("off")
            legend_ax.legend(bars, labels, title="Contributions", fontsize=10, loc="center")
            plt.tight_layout()

            figures.append((main_fig, legend_fig, category))

        return figures

    except Exception as e:
        st.error(f"❌ Erreur dans la génération des graphiques par catégorie : {str(e)}")
        return []




def plot_combined_graph_with_scenario_hatches(percentage_table, total_impact_table):
    """
    Génère des graphiques combinés par scénario avec hachures et légendes.
    Retourne une liste de tuples : (description, figure).
    """
    try:
        global contributions_colors, scenario_colors
        figures = []

        if percentage_table.empty or total_impact_table.empty:
            st.warning("Missing data for the combined scenario chart.")
            return []

        categories = percentage_table.index
        scenarios = percentage_table.columns.levels[0]
        contributions = percentage_table.columns.levels[1]
        num_scenarios = len(scenarios)
        bar_width = 0.9 / num_scenarios
        x_positions = np.arange(len(categories))

        # Définir les hachures par scénario
        hatches_list = ["//", "oo", "..", "xx", "--", "||", "++"]
        scenario_hatch_dict = {
            scenario: hatches_list[i % len(hatches_list)]
            for i, scenario in enumerate(scenarios)
        }

        # Gérer les couleurs des contributions
        cmap = plt.get_cmap("tab10")
        contrib_color_map = {}
        if contributions_colors is not None:
            if isinstance(contributions_colors, dict):
                for i, contrib in enumerate(contributions):
                    key = contrib.replace("%", "").strip().lower()
                    contrib_color_map[contrib] = contributions_colors.get(key, cmap(i % 10))
            elif hasattr(contributions_colors, "iterrows"):
                for i, contrib in enumerate(contributions):
                    key = contrib.replace("%", "").strip().lower()
                    match = contributions_colors[
                        contributions_colors["Contributions"].str.lower() == key
                    ]
                    contrib_color_map[contrib] = match["Hex Code"].iloc[0] if not match.empty else cmap(i % 10)
        else:
            for i, contrib in enumerate(contributions):
                contrib_color_map[contrib] = cmap(i % 10)

        # Gérer les couleurs des scénarios pour la légende (optionnel)
        scenario_color_map = {}
        if scenario_colors is not None:
            if isinstance(scenario_colors, dict):
                for scen in scenarios:
                    key = scen.strip().lower()
                    scenario_color_map[scen] = scenario_colors.get(key, "#FFFFFF")
            elif hasattr(scenario_colors, "iterrows"):
                for scen in scenarios:
                    key = scen.strip().lower()
                    match = scenario_colors[
                        scenario_colors["Scenario"].str.lower() == key
                    ]
                    scenario_color_map[scen] = match["Hex Code"].iloc[0] if not match.empty else "#FFFFFF"
        else:
            for scen in scenarios:
                scenario_color_map[scen] = "#FFFFFF"

        # Fonction interne pour créer un graphique
        def create_figure(show_totals=False):
            fig, ax = plt.subplots(figsize=(16, 10))
            title = "Combined Scenarios Analysis"
            title += " (with Totals)" if show_totals else ""
            ax.set_title(title, fontsize=18, pad=20, fontweight='bold')

            for i, scenario in enumerate(scenarios):
                bottom_pos = np.zeros(len(categories))
                bottom_neg = np.zeros(len(categories))

                for contrib in contributions:
                    try:
                        values = (
                            percentage_table.xs((scenario, contrib), axis=1, level=[0, 1])
                            .values.flatten()
                        )
                    except KeyError:
                        continue

                    values = np.nan_to_num(values.astype(float))
                    pos_values = np.where(values > 0, values, 0)
                    neg_values = np.where(values < 0, values, 0)

                    if np.any(pos_values):
                        ax.bar(
                            x_positions + i * bar_width,
                            pos_values,
                            bar_width,
                            bottom=bottom_pos,
                            color=contrib_color_map[contrib],
                            hatch=scenario_hatch_dict[scenario],
                            edgecolor='black'
                        )
                        bottom_pos += pos_values

                    if np.any(neg_values):
                        ax.bar(
                            x_positions + i * bar_width,
                            neg_values,
                            bar_width,
                            bottom=bottom_neg,
                            color=contrib_color_map[contrib],
                            hatch=scenario_hatch_dict[scenario],
                            edgecolor='black'
                        )
                        bottom_neg += neg_values

                # Ajouter les valeurs totales si demandé
                if show_totals:
                    scenario_clean = scenario.split("(")[-1].replace(")", "").strip()
                    try:
                        totals = total_impact_table[scenario_clean].values
                        for j, total in enumerate(totals):
                            formatted = f"{total:.2E}" if abs(total) >= 1000 else f"{total:.2f}"
                            ax.text(
                                x_positions[j] + i * bar_width,
                                bottom_pos[j] + 2,
                                formatted,
                                ha='center',
                                va='bottom',
                                rotation=90,
                                fontsize=8
                            )
                    except KeyError:
                        pass

            ax.set_xticks(x_positions + (num_scenarios - 1) * bar_width / 2)
            ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=12)
            ax.set_ylabel("Contribution (%)", fontsize=14)
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            plt.tight_layout()
            return fig

        # Graphiques principaux
        figures.append(("Main Chart", create_figure(show_totals=False)))
        figures.append(("Chart with Totals", create_figure(show_totals=True)))

        # Légende combinée
        legend_fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')

        contrib_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black')
            for _, color in contrib_color_map.items()
        ]
        scenario_handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=scenario_color_map[scenario],
                hatch=scenario_hatch_dict[scenario],
                edgecolor='black'
            )
            for scenario in scenarios
        ]

        ax.legend(
            contrib_handles + scenario_handles,
            list(contrib_color_map.keys()) + [s.split("(")[-1].replace(")", "").strip() for s in scenarios],
            title="Legend - Contributions & Scenarios",
            ncol=2,
            fontsize=8,
            loc='center'
        )
        figures.append(("Combined Legend", legend_fig))

        return figures

    except Exception as e:
        st.error(f"Error generating combined scenario chart: {str(e)}")
        return []




def generate_table_graph(df, table_name):
    """
    Generates a matplotlib figure displaying a DataFrame as a nicely formatted table:
    - Numeric values in scientific notation (2E)
    - First row (header) and first column highlighted with a lightblue background and bold text.
    """
    # Copy the DataFrame and format numeric values in 2E notation
    formatted_df = df.copy()
    formatted_df = formatted_df.applymap(
        lambda x: f"{x:.2E}" if isinstance(x, (int, float)) else x
    )

    num_rows, num_cols = formatted_df.shape
    fig_width = max(10, num_cols * 1.5)
    fig_height = max(6, num_rows * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()

    title_y_position = 0.96
    table_y_position = 0.96 - (num_rows * 0.01)
    table_y_position = max(0.70, table_y_position)

    plt.figtext(
        0.5,
        title_y_position,
        f"Table: {table_name}",
        fontsize=18,
        fontweight="bold",
        ha="center",
    )

    table = ax.table(
        cellText=formatted_df.values,
        colLabels=formatted_df.columns,
        rowLabels=formatted_df.index,
        loc="center",
        cellLoc="center",
        colWidths=[0.5] * formatted_df.shape[1],
    )

    for key, cell in table.get_celld().items():
        cell.set_fontsize(14)
        cell.set_height(0.08)

    header_color = "lightblue"
    first_col_color = "lightblue"

    # Header row
    for i in range(len(formatted_df.columns)):
        cell = table[0, i]
        cell.set_text_props(weight="bold")
        cell.set_facecolor(header_color)

    # First column
    for i in range(len(formatted_df.index)):
        cell = table[i + 1, -1]
        cell.set_text_props(weight="bold")
        cell.set_facecolor(first_col_color)

    plt.subplots_adjust(top=table_y_position, bottom=0.1)
    return fig


def find_file_path(file_name, start_directory=None):
    """
    Recursively searches for a file starting from 'start_directory'.
    If 'start_directory' is not provided, uses the user's home directory.
    """
    if start_directory is None:
        start_directory = os.path.expanduser("~")

    for root, dirs, files in os.walk(start_directory):
        if file_name in files:
            return os.path.join(root, file_name)

    return None




###############################
#           Main App
###############################

def main():
    st.title("Environmental Impact Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"], key="file_upload_main")

    if uploaded_file is not None:
        try:
            # Lire les feuilles disponibles dans le fichier
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names

            # Choix de la feuille par l'utilisateur
            selected_sheet = st.selectbox("📄 Choisissez la feuille à analyser :", sheet_names)

            # Affichage aperçu rapide (5 premières lignes)
            preview_df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, nrows=5)
            with st.expander("👁️ Aperçu de la feuille sélectionnée"):
                st.dataframe(preview_df)
            # Analyze the Excel file from a specified sheet (e.g., "Feuil2")
            initial_table, combined_table, total_impact_table, scenario_names_cleaned = analyze_excel_and_generate_tables(
                uploaded_file, sheet_name=selected_sheet
            )
            contributions_list = extract_contributions_from_initial_table(initial_table)

            # Génération des couleurs par défaut
            import matplotlib.colors as mcolors

            def generate_default_colors(names, color_list):
                return {name.lower(): color_list[i % len(color_list)] for i, name in enumerate(names)}

            default_scenario_colors = generate_default_colors(scenario_names_cleaned, list(mcolors.TABLEAU_COLORS.values()))
            default_contribution_colors = generate_default_colors(contributions_list, list(mcolors.CSS4_COLORS.values()))

            # Dictionnaires globaux modifiables
            global scenario_colors, contributions_colors
            scenario_colors = default_scenario_colors.copy()
            contributions_colors = default_contribution_colors.copy()

            # Interface de personnalisation dans la sidebar
            st.sidebar.subheader("🎨 Personnalisation des couleurs")
            customize_colors = st.sidebar.checkbox("Personnaliser les couleurs ?")

            if customize_colors:
                st.sidebar.markdown("#### Couleurs des scénarios")
                for scen in scenario_names_cleaned:
                    scen_lower = scen.lower()
                    picked = st.sidebar.color_picker(f"🎯 {scen.capitalize()}", scenario_colors[scen_lower])
                    scenario_colors[scen_lower] = picked

                st.sidebar.markdown("#### Couleurs des contributions")
                for contrib in contributions_list:
                    contrib_lower = contrib.lower()
                    picked = st.sidebar.color_picker(f"📦 {contrib}", contributions_colors[contrib_lower])
                    contributions_colors[contrib_lower] = picked

            if initial_table is not None:
                scenario_tables = generate_tables_by_scenario(initial_table, scenario_names_cleaned)
                percentage_table = generate_percentage_table(initial_table, total_impact_table)

                st.subheader("Select Charts to Generate")
                generate_scenario_comparison = st.checkbox("Generate Scenario Comparison Charts")
                generate_scenario_details = st.checkbox("Generate Detailed Scenario Charts (Vertical)")
                generate_scenario_details_horizontal = st.checkbox("Generate Detailed Scenario Charts (Horizontal)")
                generate_category_details = st.checkbox("Generate Category-by-Category Charts")
                generate_combined_charts = st.checkbox("Generate Combined Charts")

                # 1) Scenario Comparison
                if generate_scenario_comparison and total_impact_table is not None:
                    st.markdown("---")
                    st.header("Scenario Comparison")
                    comparison_figures = plot_comparison_bar_chart(total_impact_table)
                    if comparison_figures:
                        st.pyplot(comparison_figures[0])  # Main comparison
                        st.pyplot(comparison_figures[1])  # Legend
                        st.pyplot(comparison_figures[2])  # With totals

                # 2) Detailed Scenario Charts (vertical)
                if generate_scenario_details and scenario_tables:
                    st.markdown("---")
                    st.header("Detailed Analysis by Scenario (Vertical)")
                    scenario_figures = plot_relative_contribution_by_scenario(scenario_tables)
                    for idx, (main_fig, legend_fig) in enumerate(scenario_figures):
                        scenario_name = list(scenario_tables.keys())[idx]
                        st.subheader(f"Scenario: {scenario_name}")

                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.pyplot(main_fig)
                        with col2:
                            st.pyplot(legend_fig)

                        st.markdown("---")

                # 3) Detailed Scenario Charts (horizontal)
                if generate_scenario_details_horizontal and scenario_tables:
                    st.markdown("---")
                    st.header("Detailed Analysis by Scenario (Horizontal)")
                    horizontal_figures = plot_relative_contribution_by_scenario_horizontal(scenario_tables)
                    for idx, (main_fig, legend_fig) in enumerate(horizontal_figures):
                        scenario_name = list(scenario_tables.keys())[idx]
                        st.subheader(f"Scenario: {scenario_name} (Horizontal)")

                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.pyplot(main_fig)
                        with col2:
                            st.pyplot(legend_fig)

                        st.markdown("---")

                # 4) Category-by-Category Charts
                if generate_category_details and percentage_table is not None and total_impact_table is not None:
                    st.markdown("---")
                    st.header("Detailed Analysis by Impact Category")
                    category_figures = plot_stacked_bar_by_category(percentage_table, total_impact_table)
                    if category_figures:
                        for main_fig, legend_fig, category_name in category_figures:
                            st.subheader(f"Category: {category_name}")

                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.pyplot(main_fig)
                            with col2:
                                st.pyplot(legend_fig)

                            plt.close(main_fig)
                            plt.close(legend_fig)

                            st.markdown("---")

                # 5) Combined Charts
                if generate_combined_charts and percentage_table is not None and total_impact_table is not None:
                    st.markdown("---")
                    st.header("Combined View of All Scenarios")
                    combined_figures = plot_combined_graph_with_scenario_hatches(percentage_table, total_impact_table)
                    if combined_figures:
                        # Display main chart
                        for name, fig in combined_figures:
                            if name == "Main Chart":
                                st.subheader("Main Scenario Comparison")
                                st.pyplot(fig)
                                plt.close(fig)

                        # Display combined legend
                        for name, fig in combined_figures:
                            if name == "Combined Legend":
                                st.subheader("Legend")
                                st.pyplot(fig)
                                plt.close(fig)

                        # Display chart with totals
                        for name, fig in combined_figures:
                            if name == "Chart with Totals":
                                st.subheader("Scenario Comparison (with Totals)")
                                st.pyplot(fig)
                                plt.close(fig)

        except Exception as e:
            st.error(f"Error while processing the file: {str(e)}")


if __name__ == "__main__":
    main()
