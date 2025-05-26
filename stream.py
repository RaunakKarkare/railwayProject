import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

# Streamlit App
st.title("Real-Time Train Delay Optimization with Multiple RL Algorithms")
st.write("Results processed on Kaggle with GPU. Displaying decision recommendations, comparisons, and visualizations.")

# Load results
try:
    with open("results.pkl", "rb") as f:
        output = pickle.load(f)
    results_dict = output["results_dict"]
    comparison_df = output["comparison_df"]
    last_updated = output["last_updated"]
    train_numbers = output["train_numbers"]
    research_papers = output["research_papers"]
    comparison_data = output["comparison_data"]
except FileNotFoundError:
    st.error("results.pkl not found. Run the Kaggle notebook and download results.pkl.")
    st.stop()
except Exception as e:
    st.error(f"Error loading results.pkl: {e}")
    st.stop()

if not results_dict:
    st.error("No results available. Check Kaggle processing logs.")
    st.stop()

st.write(f"**Last Data Update:** {last_updated}")

# Create action decision flowchart (redefined for Streamlit)
def create_action_decision_flowchart():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    nodes = [
        {"text": "State Input\n(Parameters: Scheduled Arrival/Departure Time,\nDistance, Delay Status, Station Congestion,\nTrack Availability, Peak Hour Indicator,\nHalt Time, Original Delay)", "xy": (5, 13), "shape": "rect", "size": (4, 1.5)},
        {"text": "Generate Possible Adjustments\n(Based on Original Delay,\nStation Congestion,\nTrack Availability,\nPeak Hour Indicator)", "xy": (5, 10.5), "shape": "rect", "size": (4, 1.5)},
        {"text": "RL Policy\n(DQN/A2C/PPO: Discrete Q-Values\nSAC/DDPG: Continuous Action)", "xy": (5, 8), "shape": "rect", "size": (3, 1)},
        {"text": "Apply Action\n(Delay Adjustment)", "xy": (5, 6), "shape": "rect", "size": (3, 1)},
        {"text": "Reward Calculation\n(Delay Reduction, Congestion Penalty,\nTrack Bonus, Peak Hour Penalty,\nDownstream Impact, Increase Penalty)", "xy": (5, 4), "shape": "rect", "size": (4, 1.5)},
        {"text": "Update Delay\n(New Delay = max(0, min(Original + Adjustment)))", "xy": (5, 2), "shape": "rect", "size": (4, 1)},
        {"text": "Next Station or Done", "xy": (5, 0.5), "shape": "circle", "size": (0.5, 0.5)}
    ]
    for node in nodes:
        x, y = node["xy"]
        if node["shape"] == "rect":
            w, h = node["size"]
            ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor='lightblue', edgecolor='black'))
            ax.text(x, y, node["text"], ha='center', va='center', fontsize=8, wrap=True)
        else:
            r = node["size"][0]
            ax.add_patch(Circle((x, y), r, facecolor='lightgreen', edgecolor='black'))
            ax.text(x, y, node["text"], ha='center', va='center', fontsize=8)
    arrows = [
        ((5, 12.25), (5, 11.25)),
        ((5, 9.75), (5, 8.75)),
        ((5, 7.25), (5, 6.75)),
        ((5, 5.25), (5, 4.75)),
        ((5, 3.25), (5, 2.75)),
        ((5, 1.75), (5, 1))
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), mutation_scale=20, color='black'))
    plt.title("Action Decision Process in RL Model", fontsize=12, pad=20)
    return fig

# Train Selection
st.subheader("Train Selection")
df_optimized = results_dict.get("DQN", {}).get("df_result", pd.DataFrame())
if df_optimized.empty:
    st.warning("No DQN results available for train selection.")
    selected_train_value = train_numbers[0] if train_numbers else "16094"
else:
    unique_trains = df_optimized.drop_duplicates(subset=["Train No", "Train Name"])
    unique_trains["Train No"] = unique_trains["Train No"].astype(str)
    dropdown_options = unique_trains.apply(
        lambda row: {"label": f"{row['Train No']} - {row['Train Name']}", "value": row['Train No']},
        axis=1
    ).tolist()
    if dropdown_options:
        default_train_value = train_numbers[0] if train_numbers and train_numbers[0] in [opt["value"] for opt in dropdown_options] else dropdown_options[0]["value"]
        default_index = next((i for i, opt in enumerate(dropdown_options) if opt["value"] == default_train_value), 0)
        selected_option = st.selectbox(
            "Select Train:",
            options=dropdown_options,
            format_func=lambda opt: opt["label"],
            index=default_index
        )
        selected_train_value = selected_option["value"]
    else:
        st.warning("No trains found for selection.")
        selected_train_value = train_numbers[0] if train_numbers else "16094"

# Real-Time Decision Recommendations
st.subheader("Real-Time Decision Recommendations")
st.write("Recommended actions for railway operators based on RL algorithm outputs:")
for algo_name, result in results_dict.items():
    q_df = result.get("q_df", pd.DataFrame())
    if not q_df.empty:
        filtered_q_df = q_df[q_df["Train No"].astype(str) == selected_train_value].sort_values("SEQ")
        if not filtered_q_df.empty:
            st.write(f"**{algo_name} Decision Recommendations for Train {selected_train_value}:**")
            display_df = filtered_q_df[[
                "Station", "Station Code", "Delay Adjustment (mins)", "Recommended Decision", "Decision Rationale"
            ]].rename(columns={"Station": "Station Name"})
            st.dataframe(display_df.reset_index(drop=True))
        else:
            st.warning(f"No decision recommendations for Train {selected_train_value} ({algo_name}).")
    else:
        st.warning(f"No decision data available for {algo_name}.")

# Related Research Papers
st.subheader("Related Research Papers")
st.write("Relevant papers on RL for train delay optimization and rescheduling:")
for paper in research_papers:
    st.markdown(f"""
    **Title**: {paper['Title']}  
    **Authors**: {paper['Authors']}  
    **Publication**: {paper['Publication']}  
    **Summary**: {paper['Summary']}  
    **Relevance to Project**: {paper['Relevance']}  
    """)

# Comparison with Research Papers
st.subheader("Comparison with Research Papers")
st.write("Comparison of project’s approach with research papers:")
comparison_df_papers = pd.DataFrame(comparison_data)
st.dataframe(comparison_df_papers)
fig = go.Figure()
metrics = ["Delay Reduction (%)"]
for metric in metrics:
    fig.add_trace(go.Bar(
        x=comparison_df_papers["Approach"],
        y=comparison_df_papers[metric],
        name=metric,
        text=comparison_df_papers[metric].round(2),
        textposition='auto'
    ))
fig.update_layout(
    title="Comparison of Delay Reduction Across Approaches",
    xaxis_title="Approach",
    yaxis_title="Delay Reduction (%)",
    barmode='group',
    legend=dict(x=0.01, y=0.99)
)
st.plotly_chart(fig, use_container_width=True)

# Algorithm Comparison
st.subheader("Algorithm Comparison")
if not comparison_df.empty:
    st.write("Performance metrics for DQN, A2C, PPO, SAC, and DDPG:")
    st.dataframe(comparison_df)
    fig = go.Figure()
    metrics = ["Total Delay Reduction (min)", "Avg Optimized Delay (min)", "Percentage Reduction (%)", "Reward Variance"]
    for metric in metrics:
        fig.add_trace(go.Bar(
            x=comparison_df["Algorithm"],
            y=comparison_df[metric],
            name=metric,
            text=comparison_df[metric].round(2),
            textposition='auto'
        ))
    fig.update_layout(
        title="Comparison of RL Algorithms",
        xaxis_title="Algorithm",
        yaxis_title="Metric Value",
        barmode='group',
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No comparison data available.")

# Action Decision Process Visualization
st.subheader("Action Decision Process Visualization")
st.write("Flowchart illustrating how parameters influence action selection in the RL model:")
fig = create_action_decision_flowchart()
st.pyplot(fig)
plt.close(fig)