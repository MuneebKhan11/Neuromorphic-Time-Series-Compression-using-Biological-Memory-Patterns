from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

from sklearn.ensemble import IsolationForest
from scipy.interpolate import interp1d

st.set_page_config(page_title="NTSC-BMP Interactive App", layout='wide')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

def load_dataset(file_path):
    column_names = ['Engine Number', 'Time in Cycles',
                    'Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3'] + \
                   [f'Sensor {i}' for i in range(1, 22)]

    data = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)

    return data

def calculate_significance(data):
    sensor_columns = [col for col in data.columns if 'Sensor' in col]
    data_sensors = data[sensor_columns]
    variance = data_sensors.var(axis=1)
    significance_scores = (variance - variance.min()) / (variance.max() - variance.min())
    return significance_scores

def detect_anomalies(data):
    sensor_columns = [col for col in data.columns if 'Sensor' in col]
    data_sensors = data[sensor_columns]

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(data_sensors)

    anomaly_scores = -iso_forest.decision_function(data_sensors)
    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    return anomaly_scores

class NTSC_BMP_Compressor:
    def __init__(self, significance_threshold=0.5, anomaly_threshold=0.5,
                 consolidated_layer_retention=0.5, long_term_layer_retention=0.2):
        self.immediate_layer_retention = 1.0
        self.consolidated_layer_retention = consolidated_layer_retention
        self.long_term_layer_retention = long_term_layer_retention
        self.significance_threshold = significance_threshold
        self.anomaly_threshold = anomaly_threshold

    def compress(self, data):
        immediate_layer = data.copy()
        significance_scores = calculate_significance(data)
        anomaly_scores = detect_anomalies(data)
        consolidated_layer = self._consolidate_data(data, significance_scores)
        long_term_layer = self._retain_significant_data(data, significance_scores, anomaly_scores)

        compressed_data = {
            'Immediate Layer': immediate_layer,
            'Consolidated Layer': consolidated_layer,
            'Long-Term Layer': long_term_layer
        }

        return compressed_data, significance_scores, anomaly_scores

    def decompress(self, compressed_data, original_time_index):
        long_term_data = compressed_data['Long-Term Layer']
        if long_term_data.empty:
            return None
        reconstructed_data = self._interpolate_data(long_term_data, original_time_index)
        return reconstructed_data

    def evaluate(self, original_data, reconstructed_data, compressed_data):
        if reconstructed_data is None or reconstructed_data.empty:
            st.warning("Reconstructed data is empty. Evaluation cannot be performed.")
            return None, None

        common_times = np.intersect1d(original_data['Time in Cycles'], reconstructed_data['Time in Cycles'])
        if len(common_times) == 0:
            st.warning("No common times between original and reconstructed data. Evaluation cannot be performed.")
            return None, None

        original_aligned = original_data[original_data['Time in Cycles'].isin(common_times)]
        reconstructed_aligned = reconstructed_data[reconstructed_data['Time in Cycles'].isin(common_times)]

        mse = np.mean((original_aligned.select_dtypes(include=[np.number]).set_index('Time in Cycles') -
                       reconstructed_aligned.select_dtypes(include=[np.number]).set_index('Time in Cycles')) ** 2).mean()

        original_size = original_data.memory_usage(index=True).sum()
        compressed_size = sum(
            layer.memory_usage(index=True).sum() for layer in compressed_data.values()
        )
        compression_ratio = original_size / compressed_size  # Larger means better compression

        return mse, compression_ratio

    def _consolidate_data(self, data, significance_scores):
        threshold = np.quantile(significance_scores, 1 - self.consolidated_layer_retention)
        mask = significance_scores >= threshold
        consolidated_data = data[mask]
        return consolidated_data

    def _retain_significant_data(self, data, significance_scores, anomalies):
        significance_mask = significance_scores >= self.significance_threshold
        anomaly_mask = anomalies >= self.anomaly_threshold
        mask = significance_mask | anomaly_mask
        long_term_data = data[mask]
        return long_term_data

    def _interpolate_data(self, data, original_time_index):
        sensor_columns = [col for col in data.columns if 'Sensor' in col]
        if data['Time in Cycles'].duplicated().any():
            data = data.drop_duplicates(subset='Time in Cycles')
        interpolated_data = pd.DataFrame({'Time in Cycles': original_time_index})
        interpolated_data['Engine Number'] = data['Engine Number'].iloc[0]

        for sensor in sensor_columns:
            try:
                interp_func = interp1d(data['Time in Cycles'], data[sensor], kind='linear', fill_value="extrapolate")
                interpolated_data[sensor] = interp_func(original_time_index)
            except Exception:
                interpolated_data[sensor] = np.nan

        return interpolated_data

def main():
    st.title("🧠 Neuromorphic Time-Series Compression with NTSC-BMP (Muneeb's Algorithm)")

    st.markdown("""
    ## Welcome, Data Explorer! 👋

    **Ever wondered how our brains decide what's worth remembering?**

    Imagine you're at a carnival 🎡. The bright lights, the joyful screams, the smell of popcorn 🍿. You won't remember every single step you took, but those thrilling rides and fun moments stick with you!

    **That's exactly how Muneeb's Algorithm works with data!**

    ### What's the Adventure Today?

    - **Muneeb's Algorithm (NTSC-BMP):** A fascinating method that helps computers remember important information and summarize the rest, just like our brains do.
    - **Jet Engine Tales:** We're diving into real data from NASA's jet engines—think of it as the engines' personal diaries 📖.

    ### How Do the Engines Tell Their Stories?

    - 🛫 **Recording Every Detail:** The dataset logs sensor readings from engines during operation, like jotting down every experience.
    - 🔍 **Highlighting the Exciting Parts:** Muneeb's Algorithm picks out the big moments, such as unusual vibrations or temperature spikes.
    - ✨ **Cherishing the Memories:** Important events are kept in detail, while routine data is summarized.

    **Ready to embark on this data journey? Let's get started!**

    ---
    """)

    st.sidebar.title("🔧 Adjust the Settings")
    data_source = st.sidebar.radio(
        "Choose Data Source",
        ("Sample NASA Dataset", "Upload Your Own Dataset"),
        help="You can use our sample data or upload your own!"
    )

    if data_source == "Upload Your Own Dataset":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV or TXT file",
            type=["csv", "txt"],
            help="Make sure your file follows the required format."
        )
        if uploaded_file is not None:
            data = load_dataset(uploaded_file)
        else:
            st.sidebar.warning("Please upload a dataset to proceed.")
            st.stop()
    else:
        data_file = st.sidebar.selectbox(
            "Choose a Sample Dataset",
            ("train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt"),
            help="Different datasets represent various engine conditions."
        )
        data_path = f"data/{data_file}"
        data = load_dataset(data_path)

    st.sidebar.markdown("### 🎛️ Tune the Algorithm")
    significance_threshold = st.sidebar.slider(
        "Significance Threshold",
        0.0, 1.0, 0.5, 0.01,
        help="Higher values mean only very significant events are kept."
    )
    anomaly_threshold = st.sidebar.slider(
        "Anomaly Threshold",
        0.0, 1.0, 0.5, 0.01,
        help="Higher values mean only very unusual events are kept."
    )
    consolidated_retention = st.sidebar.slider(
        "Consolidated Layer Retention",
        0.0, 1.0, 0.5, 0.01,
        help="Higher values keep more recent details."
    )
    long_term_retention = st.sidebar.slider(
        "Long-Term Layer Retention",
        0.0, 1.0, 0.2, 0.01,
        help="Higher values keep more long-term memories."
    )

    st.sidebar.markdown("### 🎨 Visualization Options")
    unit_numbers = sorted(data['Engine Number'].unique())
    selected_unit = st.sidebar.selectbox(
        "Pick an Engine Number", unit_numbers,
        help="Select which engine's data you want to explore."
    )
    sensor_columns = [f'Sensor {i}' for i in range(1, 22)]
    selected_sensors = st.sidebar.multiselect(
        "Pick Sensors to View", sensor_columns, default=[sensor_columns[0]],
        help="Choose one or more sensors to see their data."
    )

    st.subheader("1️⃣ Exploring the Data")
    st.write("Here's a glimpse of the data we're working with:")
    st.dataframe(data.head())

    st.markdown("""
    **What's in the data?**

    - **Engine Number:** Identifier for each engine.
    - **Time in Cycles:** Number of operational cycles.
    - **Operational Settings:** Engine operating conditions.
    - **Sensors:** Measurements from various parts of the engine.

    *Feel free to explore and get familiar with the data!*
    """)

    st.markdown("### Curious to Dive Deeper?")
    if st.checkbox("Yes, let's explore more!"):
        if st.checkbox("Show Summary Statistics"):
            st.write(data.describe())

        if st.checkbox("Visualize Sensor Correlations"):
            st.write("Here's how different sensors relate to each other:")
            corr_matrix = data[sensor_columns].corr()
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=sensor_columns,
                y=sensor_columns,
                title="Sensor Correlation Heatmap",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            fig_corr.update_xaxes(side="top")
            st.plotly_chart(fig_corr, use_container_width=True)
            st.write("""
            **Interpreting the Heatmap:**

            - **Red Areas:** Sensors that increase together.
            - **Blue Areas:** When one sensor increases, the other decreases.
            - **Why it matters:** Understanding sensor relationships helps us spot patterns and anomalies!
            """)

    st.subheader("2️⃣ Unleashing Muneeb's Algorithm")
    st.write("Now, let's see the algorithm in action!")

    # Animation of data processing
    with st.spinner('Processing data...'):
        time.sleep(2)  # Simulate processing time

    # Corrected parameter name
    compressor = NTSC_BMP_Compressor(
        significance_threshold=significance_threshold,
        anomaly_threshold=anomaly_threshold,
        consolidated_layer_retention=consolidated_retention,
        long_term_layer_retention=long_term_retention  # Corrected parameter name
    )

    compressed_data, significance_scores, anomaly_scores = compressor.compress(data)

    st.success("Data compressed successfully!")

    st.markdown("""
    **How does the algorithm work?**

    - **Immediate Layer:** Like short-term memory, it retains all recent data.
    - **Consolidated Layer:** Summarizes data, focusing on important trends.
    - **Long-Term Layer:** Keeps only the most significant and unusual events.

    *It's like how we remember yesterday's highlights but only recall major events from years ago!*
    """)

    st.subheader("3️⃣ Visualizing the Engine's Story")

    unit_data = data[data['Engine Number'] == selected_unit]
    unit_time_index = unit_data['Time in Cycles'].values
    unit_significance = significance_scores[unit_data.index]
    unit_anomalies = anomaly_scores[unit_data.index]

    layer_names = ['Immediate Layer', 'Consolidated Layer', 'Long-Term Layer']
    layer_data = [compressed_data[layer][compressed_data[layer]['Engine Number'] == selected_unit] for layer in layer_names]

    for i, layer_name in enumerate(layer_names):
        st.markdown(f"#### 📂 {layer_name}")
        layer_unit_data = layer_data[i]

        if layer_unit_data.empty:
            st.write("No data is available in this layer for the selected engine. Try adjusting the sliders to retain more data.")
            continue

        st.write(f"The **{layer_name.lower()}** shows how the algorithm retains data over time.")
        fig = go.Figure()

        for sensor in selected_sensors:
            fig.add_trace(go.Scatter(
                x=layer_unit_data['Time in Cycles'],
                y=layer_unit_data[sensor],
                mode='lines+markers',
                name=sensor,
                hovertemplate=f"Time in Cycles: %{{x}}<br>{sensor}: %{{y}}<extra></extra>"
            ))

        # Add significant events
        if layer_name == 'Long-Term Layer':
            for idx in layer_unit_data['Time in Cycles']:
                fig.add_vline(x=idx, line_width=1, line_dash='dash', line_color='green')

        fig.update_layout(
            title=f"{layer_name} Data for Engine {selected_unit}",
            xaxis_title='Time in Cycles',
            yaxis_title='Sensor Readings',
            hovermode='x unified',
            legend_title="Sensors",
            transition_duration=500
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write(f"""
        **Understanding {layer_name}:**

        - **{layer_name}** helps us see how data is filtered over time.
        - **Visual Cues:** Vertical dashed lines indicate significant events.
        - **Interactive Experience:** Hover over points to see detailed information.

        *It's like watching a movie where only the key scenes are highlighted!*
        """)

        st.write("---")

    st.subheader("4️⃣ Decoding Significance and Anomalies")
    st.write("Let's explore how the algorithm assigns importance to each data point.")
    scores_df = pd.DataFrame({
        'Time in Cycles': unit_time_index,
        'Importance Score': unit_significance,
        'Unusualness Score': unit_anomalies
    })
    scores_melted = scores_df.melt(id_vars=['Time in Cycles'], var_name='Score Type', value_name='Score')
    fig_scores = px.line(
        scores_melted,
        x='Time in Cycles',
        y='Score',
        color='Score Type',
        labels={'Score': 'Score Value'},
        title='Importance and Unusualness Scores Over Time',
        markers=True
    )
    for score_type in ['Importance Score', 'Unusualness Score']:
        score_data = scores_melted[scores_melted['Score Type'] == score_type]
        if not score_data.empty:
            max_score = score_data['Score'].max()
            max_time = score_data[score_data['Score'] == max_score]['Time in Cycles'].values[0]
            fig_scores.add_annotation(
                x=max_time,
                y=max_score,
                text=f"Highest {score_type}",
                showarrow=True,
                arrowhead=2
            )

    st.plotly_chart(fig_scores, use_container_width=True)

    st.write("""
    **Interpreting the Scores:**

    - **Importance Score:** Reflects how significant a data point is.
    - **Unusualness Score:** Indicates how anomalous or unexpected a data point is.
    - **Why it matters:** Helps the algorithm decide what to remember!

    *Just like how surprising events are more memorable in our lives!*
    """)

    st.write("---")

    st.subheader("5️⃣ Measuring Compression Performance")
    reconstructed_data = compressor.decompress(compressed_data, unit_time_index)

    mse, compression_ratio = compressor.evaluate(unit_data, reconstructed_data, compressed_data)

    if mse is not None and compression_ratio is not None:
        st.write(f"**Accuracy of Reconstruction:** {mse:.2f} (Lower is better)")
        st.write(f"**Compression Efficiency:** {compression_ratio:.2f}x (Higher is better)")
        st.write("""
        **What does this mean?**

        - **Accuracy:** How closely the reconstructed data matches the original.
        - **Efficiency:** How much the data size was reduced.

        *Balancing these helps us keep important information while saving space!*
        """)
    else:
        st.write("Evaluation couldn't be performed. Try adjusting the settings to retain more data.")

    st.subheader("6️⃣ Comparing Original and Reconstructed Data")
    if reconstructed_data is not None and not reconstructed_data.empty:
        st.write("Let's visually compare the original data with the reconstructed data.")
        min_time = int(unit_data['Time in Cycles'].min())
        max_time = int(unit_data['Time in Cycles'].max())
        time_range = st.slider(
            "Select Time Range:",
            min_value=min_time,
            max_value=max_time,
            value=(min_time, max_time),
            step=1
        )
        mask_original = (unit_data['Time in Cycles'] >= time_range[0]) & (unit_data['Time in Cycles'] <= time_range[1])
        unit_data_filtered = unit_data[mask_original]

        mask_reconstructed = (reconstructed_data['Time in Cycles'] >= time_range[0]) & (
                    reconstructed_data['Time in Cycles'] <= time_range[1])
        reconstructed_data_filtered = reconstructed_data[mask_reconstructed]
        for sensor in selected_sensors:
            st.markdown(f"**Sensor: {sensor}**")
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f"Original vs. Reconstructed Data for {sensor}", f"Difference in {sensor}")
            )
            fig.add_trace(go.Scatter(
                x=unit_data_filtered['Time in Cycles'],
                y=unit_data_filtered[sensor],
                mode='lines',
                name='Original',
                line=dict(color='royalblue'),
                hovertemplate=f"Time: %{{x}}<br>Original: %{{y}}<extra></extra>"
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=reconstructed_data_filtered['Time in Cycles'],
                y=reconstructed_data_filtered[sensor],
                mode='lines',
                name='Reconstructed',
                line=dict(color='firebrick', dash='dash'),
                hovertemplate=f"Time: %{{x}}<br>Reconstructed: %{{y}}<extra></extra>"
            ), row=1, col=1)
            common_times = np.intersect1d(unit_data_filtered['Time in Cycles'],
                                          reconstructed_data_filtered['Time in Cycles'])
            if len(common_times) == 0:
                st.warning("No common times between original and reconstructed data for this sensor.")
                continue

            original_common = unit_data_filtered[unit_data_filtered['Time in Cycles'].isin(common_times)]
            reconstructed_common = reconstructed_data_filtered[
                reconstructed_data_filtered['Time in Cycles'].isin(common_times)]
            residuals = original_common[sensor].values - reconstructed_common[sensor].values
            fig.add_trace(go.Scatter(
                x=common_times,
                y=residuals,
                mode='lines+markers',
                name='Residual',
                line=dict(color='green'),
                hovertemplate=f"Time: %{{x}}<br>Residual: %{{y}}<extra></extra>"
            ), row=2, col=1)
            fig.add_hline(y=0, line_width=1, line_dash='dash', line_color='gray', row=2, col=1)
            fig.update_layout(
                height=600,
                hovermode='x unified',
                xaxis_title='',
                yaxis_title='Sensor Reading',
                xaxis2_title='Time in Cycles',
                yaxis2_title='Residual',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            st.write(f"""
            **Analyzing the Comparison:**

            - **Top Plot:** Shows how well the reconstructed data aligns with the original.
            - **Bottom Plot:** Highlights differences between the original and reconstructed data.
            - **Key Takeaway:** Smaller residuals mean better reconstruction.

            *It's like comparing a photocopy to the original document!*
            """)

            st.write("---")
    else:
        st.write("Reconstructed data is not available. Try adjusting the settings to retain more data.")

    st.subheader("7️⃣ Let's Play a Game!")
    st.write("Test your understanding with our interactive quiz!")

    # Initialize session state for quiz
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'show_retry' not in st.session_state:
        st.session_state.show_retry = False
    if 'achievements' not in st.session_state:
        st.session_state.achievements = []
    def reset_quiz():
        st.session_state.quiz_submitted = False
        st.session_state.show_retry = False
        st.rerun()

    questions = [
        {
            'question': "1️⃣ What is the main goal of Muneeb's Algorithm?",
            'options': [
                "A. To remove all data points.",
                "B. To keep important data and compress less important data.",
                "C. To increase the size of the data.",
                "D. To randomly select data points."
            ],
            'correct_answer': 1,
            'explanation': "Correct! The algorithm intelligently preserves important data while compressing less significant information."
        },
        {
            'question': "2️⃣ How does the algorithm decide which data to keep in the long-term layer?",
            'options': [
                "A. Based on random selection.",
                "B. By selecting data points with high importance or unusualness scores.",
                "C. By keeping every tenth data point.",
                "D. By user manual selection."
            ],
            'correct_answer': 1,
            'explanation': "You're right! It uses significance and anomaly scores to retain key data points."
        },
        {
            'question': "3️⃣ What does a higher compression efficiency indicate?",
            'options': [
                "A. The data was not compressed much.",
                "B. The data was compressed more.",
                "C. The data quality improved.",
                "D. The algorithm failed."
            ],
            'correct_answer': 1,
            'explanation': "Exactly! A higher efficiency means more data was compressed effectively."
        },
        {
            'question': "4️⃣ Why is the reconstructed data sometimes different from the original?",
            'options': [
                "A. Because the algorithm adds random noise.",
                "B. Because some data is lost during compression.",
                "C. Because the sensors malfunctioned.",
                "D. Because of a glitch in the system."
            ],
            'correct_answer': 1,
            'explanation': "Spot on! Compression involves trade-offs, so some data differences are expected."
        }
    ]
    if not st.session_state.quiz_submitted:
        with st.form(key='quiz_form'):
            user_answers = []

            for i, q in enumerate(questions):
                user_answer = st.radio(
                    q['question'],
                    q['options'],
                    key=f'question_{i}'
                )
                user_answers.append(user_answer)

            submit_button = st.form_submit_button(label='Submit Answers')

            if submit_button:
                st.session_state.quiz_submitted = True
                st.session_state.user_answers = user_answers
                st.rerun()

    if st.session_state.quiz_submitted:
        score = 0
        all_correct = True

        for i, (user_answer, question) in enumerate(zip(st.session_state.user_answers, questions)):
            correct_answer = question['options'][question['correct_answer']]

            if user_answer == correct_answer:
                st.success(f"""
                ✅ Question {i + 1}: {question['explanation']}
                """)
                score += 1
            else:
                all_correct = False
                st.error(f"""
                ❌ Question {i + 1}: The correct answer is: {correct_answer}

                {question['explanation']}
                """)

        if all_correct:
            st.balloons()
            st.session_state.achievements.append("Quiz Master")
            st.success(f"""
            🏆 Congratulations! You scored {score}/{len(questions)}!

            You've earned the **Quiz Master** badge!

            **Your Achievements:** {', '.join(st.session_state.achievements)}
            """)
        else:
            percentage = (score / len(questions)) * 100

            st.info(f"""
            You scored {score}/{len(questions)} ({percentage:.0f}%)

            **Your Achievements:** {', '.join(st.session_state.achievements)}
            """)

            st.button("Try Again", on_click=reset_quiz)
    st.markdown("""
        ---

        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin: 2rem 0;'>
            <h2 style='color: white; margin-bottom: 1rem;'>Have Questions? We've Got Answers!</h2>
            <p style='color: white; opacity: 0.9; margin-bottom: 1.5rem;'>Explore our comprehensive FAQ section to learn more about NTSC-BMP.</p>
            <a href="/faq" style='display: inline-block; background: white; color: #667eea; padding: 0.75rem 1.5rem; border-radius: 5px; text-decoration: none; font-weight: bold; transition: transform 0.2s ease;'>
                View FAQ →
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ---

    **Feel free to adjust the settings and explore different scenarios!**

    **Fun Fact:** Our brains process around 70,000 thoughts per day. Muneeb's Algorithm helps us understand how to manage such vast amounts of data efficiently!

    **Thank you for joining this data adventure!**

    ### Connect with Us

    - **GitHub:** [MuneebKhan11](https://github.com/MuneebKhan11)
    - **LinkedIn:** [Muneeb Khan](https://www.linkedin.com/in/khanmuneeb786/)

    *Happy Exploring!* 🚀
    """)

if __name__ == '__main__':
    main()
